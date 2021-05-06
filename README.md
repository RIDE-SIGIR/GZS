# GZS-IntentDetection
Generalized Zero-Shot Intent Detection via Commonsense Knowledge.


0. Download datasets from:

i. SNIPS: https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/tree/master/data/snips_Intent_Detection_and_Slot_Filling

ii. SGD: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

iii. MultiWoz_2.2: https://github.com/budzianowski/multiwoz/tree/master/data
and place in 'raw_data' or whatever directory you plan to use in the preprocessing code.

iv. ConceptNet KG: https://github.com/commonsense/conceptnet5/wiki/Downloads
and palce it in 'KG_conceptnet' or whatever directory you plan to use in the kg preprocessing code.


1. Preprocessing the datasets

i. Datasets preprocessing for SNIPS, SGD, MultiWoz2.2
run: preprocessing/extract_intent_utterances.py -dataset (one value from  snips, dstc8-sgd, MultiWOZ_2.2)

ii. Dataset splits for train, validation, seen/unseen/general testing and negative sampling
run: preprocessing/data_split_neg_sampling.py -dataset (one value from  snips, dstc8-sgd, MultiWOZ_2.2)


2. Link prediction for meta-features generation

i. preprocess the knowledge graph for training the link prediction model.
run: SimplE/kg_preprocess.py

ii. Pre-train the link predictor model and generate the meta-features
run: SimplE/main.py


3. Train the PU classifier that is used at inference time

i. data split for seen(positive) and unseen(unlabelled)
run: pu/data_split_pu.py

ii. train PU classifier
run: pu/pu_classifier.py


4. Train the core model for intent detection

run: intent_model/model.py


Dependencies:
1. NLTK >= 3.4.5 
2. Spacy >= 2.2.1
3. Pytorch >=1.3.0
4. TorchText >= 0.4.0
5. sklearn >=0.19
6. tensorflow_hub >= 0.7.0
7. pulearn >= 0.0.3
8. Numpy >= 1.17.3
9. python >=3.7.4
10. ConceptNet  >= 5.6.0
11. wordfreq >= 2.3.2
