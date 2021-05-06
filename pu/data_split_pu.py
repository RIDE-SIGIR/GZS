import argparse
import json
import os

def get_parameter():
    parser = argparse.ArgumentParser(description='DSTC8-SGD: Dataset split')

    parser.add_argument('-path', action="store", default="../raw_data/")
    parser.add_argument('-outpath', action="store", default="../data/")
    parser.add_argument('-dataset', action="store", default="dstc8-sgd")

    args = parser.parse_args()

    return args

#utils
def get_intent_utterances(path, data_type):
    sen_intent = []

    for train in data_type:
        for file_index in range(1, 128):
            f_name = '{:03d}'.format(file_index)
            file_path = path+train+"/dialogues_"+f_name+".json"
            print(file_path)
            if not os.path.exists(file_path):
                continue
            
            with open(file_path) as f:
                data = json.load(f)
            
                for item in data:
                    if "turns" in item:
                        turns = item["turns"]
                    
                        cur_intent = None
                        for turn in turns:

                            if "speaker" in turn and "utterance" in turn:
                                utterance = turn["utterance"].replace('\n', '').replace(r'\n', ' ').replace(r'\r', '').replace('  ', ' ')

                                if "frames" in turn:
                                    frame = turn["frames"][0]
                                
                                    # user intent
                                    if "slots" in frame and "state" in frame and "active_intent" in frame["state"]: 
                                        
                                        active_intent = frame["state"]["active_intent"]
                                        
                                        if cur_intent !=active_intent and active_intent != "NONE":
                                            cur_intent = active_intent
                                            sen_intent.append(utterance+"\t"+active_intent)
                                    
                                    # system offer intent
                                    elif "actions" in frame:
                                        action = frame["actions"][0]
                                        if "act" in action and action["act"] == "OFFER_INTENT" and "canonical_values" in action and len(action["canonical_values"]) > 0:
                                                
                                            active_intent = action["canonical_values"][0]
              
                                            if cur_intent !=active_intent and active_intent != "NONE":
                                                    
                                                cur_intent = active_intent
                                                sen_intent.append(utterance+"\t"+active_intent)
    return list(set(sen_intent))                                      


def split_seen_unseen(mylist, pos):
    seen,unseen = [], []
    for line in mylist:
        utterance, intent = line.split("\t")

        if intent in pos:
            seen.append(utterance)
            
        else:
            unseen.append(utterance)

    return seen, unseen

def write_to_file(fname, mylist):
    f = open(fname, "w")
    for line in mylist:
        f.write(line+"\n")
    f.close()

if __name__ == '__main__':
    args = get_parameter()
    path = args.path
    outpath = args.outpath
    dataset = args.dataset

    train_utterances = get_intent_utterances(path+dataset+"/", ["train"])
    valid_utterances = get_intent_utterances(path+dataset+"/", ["dev"])
    test_utterances = get_intent_utterances(path+dataset+"/", ["test"])


    # currently it is set up for SGD as standard splits
    pos = list(set([line.split("\t")[1] for line in train_utterances if len(line.split("\t")) == 2])) # seen/training intents
    train_utterances = [line.split("\t")[0] for line in train_utterances]
    valid_utterances = [line.split("\t")[0] for line in valid_utterances]

    test_seen, test_unseen = split_seen_unseen(test_utterances, pos)

    print("# pos utterances", len(train_utterances))
    print("# unlabelled utterances", len(valid_utterances))
    

    write_to_file(outpath+dataset+"/pu_pos.txt", train_utterances)
    write_to_file(outpath+dataset+"/pu_unlabelled.txt", valid_utterances)
    write_to_file(outpath+dataset+"/pu_test_pos.txt", test_seen)
    write_to_file(outpath+dataset+"/pu_test_neg.txt", test_unseen)

    print("dataset saved to", outpath+dataset)

