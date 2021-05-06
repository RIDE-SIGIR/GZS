import torch
from dataset import Dataset
import numpy as np
from measure import Measure
from os import listdir
from os.path import isfile, join

class Scorer:
    def __init__(self, dataset, model_path, valid_or_test):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location = self.device)
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())

    def get_rank(self, sim_scores):#assuming the test fact is the first one
        return (sim_scores >= sim_scores[0]).sum()

    def create_queries(self, fact, head_or_tail):
        head, rel, tail = fact
        if head_or_tail == "head":
            return [(i, rel, tail) for i in range(self.dataset.num_ent())]
        elif head_or_tail == "tail":
            return [(head, rel, i) for i in range(self.dataset.num_ent())]

    def add_fact_and_shred(self, fact, queries, raw_or_fil):
        if raw_or_fil == "raw":
            result = [tuple(fact)] + queries
        elif raw_or_fil == "fil":
            result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)

        return self.shred_facts(result)
    
    def score(self, triple): 
        h,r,t = triple
        if r in self.dataset.rel2id and h in self.dataset.ent2id and t in self.dataset.ent2id:
            h,r,t = self.dataset.triple2ids(triple)
            h = torch.LongTensor([h]).to(self.device)
            r = torch.LongTensor([r]).to(self.device)
            t = torch.LongTensor([t]).to(self.device)
            sim_scores = self.model(h, r, t).cpu().data.numpy()[0]
            return min(1, max(0, sim_scores))

        else:
            return 0.0

    def shred_facts(self, triples):
        heads  = [triples[i][0] for i in range(len(triples))]
        rels   = [triples[i][1] for i in range(len(triples))]
        tails  = [triples[i][2] for i in range(len(triples))]
        return torch.LongTensor(heads).to(self.device), torch.LongTensor(rels).to(self.device), torch.LongTensor(tails).to(self.device)

    def allFactsAsTuples(self):
        tuples = []
        for spl in self.dataset.data:
            for fact in self.dataset.data[spl]:
                tuples.append(tuple(fact))
        
        return tuples
