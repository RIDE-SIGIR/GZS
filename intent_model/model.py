import tempfile

from overrides import overrides

import torch
import torch.optim as optim
import numpy as np

from torch.nn.modules.linear import Linear

import allennlp

from allennlp.common.checks import check_dimensions_match, ConfigurationError

from allennlp.models import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions


from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import sequence_cross_entropy_with_logits

from typing import Dict, Optional, Iterable, List, Tuple, Any, cast

from allennlp.data import DataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import SequenceLabelField, TextField, LabelField, ArrayField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.samplers import BasicBatchSampler, BucketBatchSampler, SequentialSampler

from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure

import allennlp.nn.util as util
from allennlp.training.util import evaluate

from allennlp.modules.seq2vec_encoders import CnnEncoder, PytorchSeq2VecWrapper

from allennlp.modules.token_embedders import (
    Embedding,
    TokenCharactersEncoder,
    ElmoTokenEmbedder,
    PretrainedTransformerEmbedder,
    PretrainedTransformerMismatchedEmbedder,
)

from allennlp.data.tokenizers import (
    CharacterTokenizer,
    PretrainedTransformerTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
)
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import ReduceOnPlateauLearningRateScheduler

from allennlp.modules.matrix_attention import (
    LinearMatrixAttention, 
    MatrixAttention
)

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import util

from allennlp.nn import Activation

from allennlp.nn.util import replace_masked_values, min_value_of_dtype

from allennlp.nn import InitializerApplicator

@DatasetReader.register('pair-classification')
class IOBDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer_space = WhitespaceTokenizer()
        self.tokenizer_spacy = SpacyTokenizer(language = "en_core_web_md", 
                                              pos_tags = True, split_on_spaces = True)
        self.token_indexers = {
            'elmo_tokens': ELMoTokenCharactersIndexer(),
            'token_characters': TokenCharactersIndexer(namespace='character_vocab',
                                                      min_padding_length=2),
            'pos_tags': SingleIdTokenIndexer(namespace='pos_tag_vocab', default_value='NNP',
                                     feature_name='tag_')
        } 
        
        self.intent_indexers = {
            'elmo_tokens': ELMoTokenCharactersIndexer(),
            'token_characters': TokenCharactersIndexer(namespace='character_vocab',
                                                      min_padding_length=2),
            'pos_tags': SingleIdTokenIndexer(namespace='pos_tag_vocab', default_value='NNP',
                                     feature_name='tag_')
        }
        
        
    def text_to_instance(self, tokens: List[Token], intent: List[Token], 
                         rmf: str = None,
                         label: int = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        intent_field = TextField(intent, self.intent_indexers)
        
        
        fields = {"utterance": sentence_field, 
                  "intent": intent_field
                 }

        if label:
            fields["label"] = LabelField(label)
            
        if rmf:
            rmf = np.fromstring(rmf, dtype=float, sep=' ')
            fields["rmf"] = ArrayField(rmf)
            
        
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            for line in f:
                sentence, intent, rmf, label = line.strip().split('\t')
                yield self.text_to_instance(self.tokenizer_space.tokenize(sentence),
                                            self.tokenizer_space.tokenize(intent),
                                            rmf,
                                            label
                                            )


@Model.register("matcher")
class Matcher(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        rmf_layer: FeedForward,
        classifier_feedforward: FeedForward,
        dropout: float = 0.3,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        self.rmf_layer = rmf_layer
        self.classifier_feedforward = classifier_feedforward

        self.dropout = torch.nn.Dropout(dropout)

        self.metrics = {"accuracy": CategoricalAccuracy()}

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        utterance: TextFieldTensors,
        intent: TextFieldTensors,
        rmf: torch.Tensor, 
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        

        mask_utterance = util.get_text_field_mask(utterance)
        mask_intent = util.get_text_field_mask(intent)

        # embedding and encoding of the utterance
        embedded_utterance = self.dropout(self.text_field_embedder(utterance))
        encoded_utterance = self.dropout(self.encoder(embedded_utterance, mask_utterance))
        
        # embedding and encoding of the intent
        embedded_intent = self.dropout(self.text_field_embedder(intent))
        encoded_intent = self.dropout(self.encoder(embedded_intent, mask_intent))
        
        rmf = self.rmf_layer(rmf)

        concat_features = torch.cat([encoded_utterance, encoded_intent, rmf], dim=-1)
        
        # the final forward layer
        logits = self.classifier_feedforward(concat_features)
        
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Converts indices to string labels, and adds a `"label"` key to the result.
        """
        predictions = output_dict["probs"].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output_dict["label"] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }

    default_predictor = "textual_entailment"



def build_dataset_reader() -> DatasetReader:
    return IOBDatasetReader()


def read_data(reader: DatasetReader) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    
    training_data = reader.read('../data/snips/utterances_train_features.txt')
    validation_data = reader.read('../data/snips/utterances_valid_features.txt')
    
    training_data = AllennlpDataset(training_data)
    validation_data = AllennlpDataset(validation_data)
    
    print("train:",len(training_data), "validation:", len(validation_data))
    return training_data, validation_data

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    
    EMBEDDING_DIM = 300
    ELMO_DIM = 1024
    NUM_FILTERS = 60
    NGRAM_FILTER_SIZES = (2, 3, 4, 5, 6)
    #out_dim for char = len(NGRAM_FILTER_SIZES) * NUM_FILTERS
    
    HIDDEN_DIM = 300
    F_OUT1 = 900
    F_OUT2 = 200
    F_OUT = 2
    RMF_DIM = 140
    RMF_DIM_OUT = 100
    
    
    elmo_options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    elmo_weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    
    elmo_embedding = ElmoTokenEmbedder(options_file=elmo_options_file,
                                   weight_file=elmo_weight_file)

    
    # This is for encoding the characters in each token.
    character_embedding = Embedding(vocab = vocab,
                                    embedding_dim = EMBEDDING_DIM,
                                    vocab_namespace = 'character_vocab'
                                )
    cnn_encoder = CnnEncoder(embedding_dim=EMBEDDING_DIM, 
                             num_filters=NUM_FILTERS, 
                             ngram_filter_sizes = NGRAM_FILTER_SIZES
                            )
    token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

    # This is for embedding the part of speech tag of each token.
    pos_tag_embedding = Embedding(vocab=vocab, 
                                  embedding_dim=EMBEDDING_DIM,
                                  vocab_namespace='pos_tag_vocab'
                                 )
    
    text_embedder = BasicTextFieldEmbedder(token_embedders={'elmo_tokens': elmo_embedding,
                                                       'token_characters': token_encoder,
                                                       'pos_tags': pos_tag_embedding
                                                      })
    
    ##encoder
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM( EMBEDDING_DIM + ELMO_DIM + len(NGRAM_FILTER_SIZES) * NUM_FILTERS ,
                                               HIDDEN_DIM, 
                                               num_layers = 2,
                                               batch_first=True,
                                               bidirectional = True
                                               ))
    ## FF to combines two lstm inputs
    final_linear_layer = FeedForward(HIDDEN_DIM * 4 + RMF_DIM_OUT, 
                                3, 
                                [F_OUT1, F_OUT2, F_OUT], 
                                torch.nn.ReLU(),
                                0.3
                                )

    ## FF to combines two lstm inputs
    rmf_linear_layer = FeedForward(RMF_DIM, 
                                1, 
                                RMF_DIM_OUT, 
                                torch.nn.Sigmoid(),
                                0.3
                                )
    #Matching model
    model = Matcher(vocab = vocab,
                  text_field_embedder = text_embedder, 
                  encoder = encoder,
                  rmf_layer = rmf_linear_layer,
                  classifier_feedforward = final_linear_layer
                 )
    
    return model


def build_data_loaders(train_data: torch.utils.data.Dataset,
                       dev_data: torch.utils.data.Dataset,
                      ) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False)
    return train_loader, dev_loader


def build_trainer(model: Model, serialization_dir: str, train_loader: DataLoader,
                  dev_loader: DataLoader) -> Trainer:
    parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr = 0.001)
    scheduler = ReduceOnPlateauLearningRateScheduler(optimizer = optimizer,
                                                     patience = 5,
                                                    verbose=True)
    
    trainer = GradientDescentTrainer(model=model, serialization_dir=serialization_dir, 
                                     data_loader=train_loader, validation_data_loader=dev_loader, 
                                     learning_rate_scheduler = scheduler,
                                     patience=20, num_epochs=200,
                                     optimizer=optimizer,
                                    validation_metric = "+accuracy",
                                    )
    return trainer

def run_training_loop():
    dataset_reader = build_dataset_reader()

    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    print(vocab)
    model = build_model(vocab)

    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    
    train_loader, dev_loader = build_data_loaders(train_data, dev_data)

    serialization_dir = "../etc/match/"
    
    trainer = build_trainer(
            model,
            serialization_dir,
            train_loader,
            dev_loader
        )
    trainer.train()
        
    #save vocab
    vocab.save_to_files(serialization_dir+"vocabulary")
    
    # Here's how to save the model.
    with open(serialization_dir+"model_zs.pt", 'wb') as f:
        torch.save(model.state_dict(), f)
    
    return model, dataset_reader

model, dataset_reader = run_training_loop()
