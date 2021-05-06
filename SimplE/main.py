from trainer import Trainer
from tester import Tester
from scorer import Scorer
from dataset import Dataset
import argparse
import time

from nltk import ngrams
import random
import numpy as np

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=1, type=int, help="number of epochs")
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="conceptnet", type=str, help="wordnet dataset")
    parser.add_argument('-emb_dim', default=200, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=10, type=int, help="number of negative examples per positive example")
    parser.add_argument('-batch_size', default=1415, type=int, help="batch size")
    parser.add_argument('-save_each', default=1, type=int, help="validate every k epochs")

    parser.add_argument('-path', action="store", default="../data/snips/")

    args = parser.parse_args()
    return args
    
def getindices(s):
    return [i for i, c in enumerate(s) if c.isupper()]

def find_ngram(sentence, n = 4):
    grams = set()
    for index in range(1,n+1):
        items = ngrams(sentence.split(), index)
        for item in items:
            grams.add(item)


    return ["_".join(item).lower() for item in grams]

def find_intent(intent):
    split = getindices(intent[1:])
    action = intent[:split[0]+1]

    objs = []
    if len(split) >= 2:
        objs.append(intent[ split[0]+1 : split[1]+1 ] )
        objs.append(intent[ split[1]+1 : ])
    else:
        objs.append(intent[split[0]+1 : ])
    return action, objs
def get_score(myscorer,triple):
    if triple[1] == "self_rel" and triple[0] == triple[2]:
        return 1.0
    return myscorer.score(triple)


def get_meta_features(myscorer, sentence, intent):
    words = find_ngram(sentence)
    intent = find_intent(intent)

    relations = ["self_rel", "/r/RelatedTo", "/r/FormOf", "/r/IsA", "/r/PartOf", "/r/HasA", "/r/UsedFor", 
                "/r/CapableOf", "/r/AtLocation", "/r/Causes", "/r/HasSubevent", "/r/HasFirstSubevent",
                "/r/HasLastSubevent" , "/r/HasPrerequisite", "/r/HasProperty", "/r/MotivatedByGoal",
                "/r/ObstructedBy", "/r/Desires", "/r/CreatedBy", "/r/Synonym", "/r/Antonym", "/r/DistinctFrom",
                "/r/DerivedFrom", "/r/SymbolOf", "/r/DefinedAs", "/r/MannerOf", "/r/LocatedNear", "/r/HasContext",
                "/r/SimilarTo", "/r/EtymologicallyRelatedTo", "/r/EtymologicallyDerivedFrom", "/r/CausesDesire",
                "/r/MadeOf", "/r/ReceivesAction", "/r/ExternalURL"]

    #with action
    features = []
    action = intent[0].lower()
    for rel in relations:
        features.append(max([get_score(myscorer,["/c/en/"+word,rel,"/c/en/"+action]) for word in words]))
        features.append(max([get_score(myscorer,["/c/en/"+action,rel,"/c/en/"+word]) for word in words]))


    # with objects
    obj = intent[1]
    if len(obj) >=2:
        for rel in relations:
            t1 = np.array([get_score(myscorer,["/c/en/"+word,rel,"/c/en/"+obj[0].lower()]) for word in words])
            t2 = np.array([get_score(myscorer,["/c/en/"+word,rel,"/c/en/"+obj[1].lower()]) for word in words])
            t = np.true_divide(np.add(t1, t2), 2)
            features.append(max(t.tolist()))

            t3 = np.array([get_score(myscorer,["/c/en/"+obj[0].lower(),rel,"/c/en/"+word]) for word in words])
            t4 = np.array([get_score(myscorer,["/c/en/"+obj[1].lower(),rel,"/c/en/"+word]) for word in words])
            tt = np.true_divide(np.add(t3, t4), 2)
            features.append(max(tt.tolist()))   
    else:
        for rel in relations:
            features.append(max([get_score(myscorer,["/c/en/"+word,rel,"/c/en/"+obj[0].lower()]) for word in words]))
            features.append(max([get_score(myscorer,["/c/en/"+obj[0].lower(),rel,"/c/en/"+word]) for word in words]))
    return features


def write_to_file(fname, mylist):
    f = open(fname, "w")
    for line in mylist:
        f.write(line+"\n")
    f.close() 

def prepare_data(myscorer, path):
    lines = [line.strip() for line in open(path, 'r')]

    data = []
    for line in lines:
        utterance, intent, label = line.split("\t")
        features = get_meta_features(myscorer, utterance, intent)
        data.append(utterance+"\t"+intent+"\t"+" ".join(str(v) for v in features)+"\t"+str(label))

    return data

if __name__ == '__main__':
    args = get_parameter()
    dataset = Dataset(args.dataset)

    print("~~~~ Training ~~~~")
    trainer = Trainer(dataset, args)
    trainer.train()

    print("~~~~ Select best epoch on validation set ~~~~")
    epochs2test = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
    dataset = Dataset(args.dataset)
    
    best_mrr = -1.0
    best_epoch = "0"
    for epoch in epochs2test:
        start = time.time()
        print(epoch)
        model_path = "models/" + args.dataset + "/" + epoch + ".chkpnt"
        tester = Tester(dataset, model_path, "valid")
        mrr = tester.test()
        if mrr > best_mrr:
            best_mrr = mrr
            best_epoch = epoch
        print(time.time() - start)

    print("Best epoch: " + best_epoch)

    print("~~~~ scoring on the best epoch ~~~~")
    best_model_path = "models/" + args.dataset + "/" + best_epoch + ".chkpnt"
    myscorer = Scorer(dataset, best_model_path, "test")

    path = args.path

    train_data = prepare_data(myscorer, path+"/train.txt")
    valid_data = prepare_data(myscorer, path+"/valid.txt")
    test_data = prepare_data(myscorer, path+"/test.txt")

    write_to_file(path+"utterances_train_features.txt", train_data)
    write_to_file(path+"utterances_valid_features.txt", valid_data)
    write_to_file(path+"utterances_test_features.txt", test_data)
