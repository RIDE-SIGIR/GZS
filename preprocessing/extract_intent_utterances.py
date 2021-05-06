import argparse
import json
import os

def get_parameter():
    parser = argparse.ArgumentParser(description='DSTC8-SGD extract utterances with intents')

    parser.add_argument('-input_path', action="store", default="../raw_data/")
    parser.add_argument('-out_path', action="store", default="../data/")
    parser.add_argument('-dataset', action="store", default="dstc8-sgd")

    args = parser.parse_args()
    return args

def get_intent_utterances_sgd(path):
    data_type = ["train", "dev", "test"]
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


def get_intent_utterances_multiwoz(path):
    data_type = ["train", "dev", "test"]
    
    sen_intent = []

    for train in data_type:
        for file_index in range(1, 18):
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

                            #utterance (user)
                            if "speaker" in turn and "utterance" in turn: # and turn["speaker"] == "USER"
                                utterance = turn["utterance"]

                                if "frames" in turn:
                                    frames = turn["frames"]
                                    for frame in frames:
                                    
                                        if "state" in frame and "active_intent" in frame["state"] and frame["state"]["active_intent"] != "NONE" :
                                        
                                            active_intent = frame["state"]["active_intent"]
                                        
                                            if cur_intent !=active_intent and active_intent != "NONE":
                                                cur_intent = active_intent
                                                sen_intent.append(utterance+"\t"+active_intent)

    return list(set(sen_intent))                                      


def get_intent_utterances_snips(path):
    data_types = ["train", "valid", "test"]
    all_lines = []
    for data in data_types:
        in_lines = open(path+data+"/seq.in", "r").readlines()
        all_lines.extend([line.strip() for line in in_lines if len(line) > 0 ])

    all_labels = []
    for data in data_types:
        in_lines = open(path+data+"/label", "r").readlines()
        all_labels.extend([line.strip() for line in in_lines if len(line) > 0 ])

    #zip utterance and labels
    return list(set(['\t'.join(map(str, i)) for i in zip(all_lines, all_labels)]))


def write_intents(path, sen_intent):
    if not os.path.exists(path):
        os.makedirs(path)

    f = open(path+"all_intents.txt", "w")
    for line in sen_intent:
        f.write(line+"\n")
        
    f.close()                
                                        
if __name__ == '__main__':
    args = get_parameter()
    input_path = args.input_path
    out_path = args.out_path
    dataset = args.dataset

    if dataset == "dstc8-sgd":
        all_intent_sentences = get_intent_utterances_sgd(input_path+dataset+"/")
    elif dataset == "MultiWOZ_2.2":
        all_intent_sentences = get_intent_utterances_multiwoz(input_path+dataset+"/")
    elif dataset == "snips":
        all_intent_sentences = get_intent_utterances_snips(input_path+dataset+"/")
    else:
        print(dataset,"not supported yet")

    print("total utterances with intents:", len(all_intent_sentences))
    write_intents(out_path+dataset+"/", all_intent_sentences)
    print("results written to:", out_path+dataset)
    