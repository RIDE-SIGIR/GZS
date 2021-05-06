import pickle, json, requests, csv, copy, os, re
from tqdm import tqdm

import numpy as np
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import urllib.request, urllib.parse
from text_to_uri import *

import spacy
nlp = spacy.load('en_core_web_sm')
import argparse

def get_parameter():
    parser = argparse.ArgumentParser(description='KG data preprocessing')
    parser.add_argument('-path', action="store", default="../KG_conceptnet/")
    parser.add_argument('-outpath', action="store", default="datasets/conceptnet/")
    args = parser.parse_args()

    return args

def remove_word_sense(sub):
    if sub.count('/') > 3:
        if sub.count('/') > 4:
            print(sub)
            assert False, "URI error (with more than 4 slashes)"
        sub = sub[:sub.rfind('/')]
    return sub

def lemmatise_ConceptNet_label(label):
    if '_' in label:
        return label
    else:
        tag = nltk.pos_tag([label])[0][1]
        if tag not in pos_dict:
            return label
        else:
            return lemmatizer.lemmatize(label, pos_dict[tag])

def read_all_nodes(filename): # get all distinct uri in conceptnet (without part of speech)
    nodes = set()
    
    with open(filename, 'r', encoding = "utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for line in tqdm(reader):
            if not line[2].startswith('/c/en/') or not line[3].startswith('/c/en/') or \
            not line[1] in relationsips: 
                # only relationships with english nodes
                continue
            # discard with weight < 1    
            details = json.loads(line[4])
            if details['weight'] < 1.0:
                continue
                    
            sub = lemmatise_ConceptNet_label(remove_word_sense(line[2]))
            obj = lemmatise_ConceptNet_label(remove_word_sense(line[3]))
            predicate = line[1]
            
            nodes.add( (sub, predicate, obj) )
            
    return nodes

def write_to_file(fname, nodes):
    f = open(fname, "w")
    for line in nodes:
        sub, predicate, obj = line
        f.write(sub+"\t"+predicate+"\t"+obj+"\n")
    f.close()

if __name__ == '__main__':
    args = get_parameter()

    path = args.path

    relationsips = ["/r/RelatedTo", "/r/FormOf", "/r/IsA", "/r/PartOf", "/r/HasA", "/r/UsedFor", 
                "/r/CapableOf", "/r/AtLocation", "/r/Causes", "/r/HasSubevent", "/r/HasFirstSubevent",
                "/r/HasLastSubevent" , "/r/HasPrerequisite", "/r/HasProperty", "/r/MotivatedByGoal",
                "/r/ObstructedBy", "/r/Desires", "/r/CreatedBy", "/r/Synonym", "/r/Antonym", "/r/DistinctFrom",
                "/r/DerivedFrom", "/r/SymbolOf", "/r/DefinedAs", "/r/MannerOf", "/r/LocatedNear", "/r/HasContext",
                "/r/SimilarTo", "/r/EtymologicallyRelatedTo", "/r/EtymologicallyDerivedFrom", "/r/CausesDesire",
                "/r/MadeOf", "/r/ReceivesAction"]

    pos_dict = {'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
           'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n',
           'RB': 'r', 'RBR': 'r', 'RBS': 'r',
           'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v'}

    lemmatizer = WordNetLemmatizer()


    all_nodes = read_all_nodes(path+"conceptnet-assertions-5.6.0.csv")
    print(len(all_nodes))


    train,dev,test = set(), set(), set()
    for line in all_nodes:
        r = random.uniform(0, 1)
        if r <=0.85:
            train.add(line)
        elif r <=0.95:
            dev.add(line)
        else:
            test.add(line)

write_to_file(args.outpath+"train.txt", train)
write_to_file(args.outpath+"valid.txt", dev)
write_to_file(args.outpath+"test.txt", test)

print("preprocessed KG written to", args.outpath)
