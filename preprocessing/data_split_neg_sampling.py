import argparse
import random

def get_parameter():
    parser = argparse.ArgumentParser(description='DSTC8-SGD: Dataset split')

    parser.add_argument('-path', action="store", default="../data/")
    parser.add_argument('-dataset', action="store", default="dstc8-sgd")
    parser.add_argument('-numseen', type=int, action="store", default=12)

    args = parser.parse_args()

    return args

#utils
def get_utterances(path):
    lines = open(path+"all_intents.txt", "r").readlines()
    lines = [line.strip() for line in lines]
    return lines

def divide_by_intent(lines):
    all_lines = {}
    for line in lines:
        intent = line.split("\t")[1]
        if intent not in all_lines:
            all_lines[intent] = []
        all_lines[intent].append(line)
    return all_lines

def split_data(all_lines, seen, unseen):
    train,valid,test = [],[],[]
    for intent, lines in all_lines.items():
        if intent in seen:
            train.extend(lines)
        elif intent in unseen:
            valid_num = int(len(lines) * 0.2)
            valid.extend(lines[:valid_num])
            test.extend(lines[valid_num:])
        else:
            print("something went wrong")
    return train, valid, test


def getindices(s):
    return [i for i, c in enumerate(s) if c.isupper()]

def neg_samples(data, intents, dataset, is_test=False):
    neg_data = []
    for line in data:
        utterance, intent = line.split("\t") 
        #positive
        neg_data.append(utterance+"\t"+intent+"\t"+str(1))

        #negative samples
        if dataset == "MultiWOZ_2.2":
            action, obj = intent.split("_")
        else:
            split = getindices(intent[1:])[0]+1
            action = intent[:split]
            obj = intent[split:]
        
        neg_1 = [s_intent for s_intent in intents if action in s_intent]
        neg_1 = list(set(neg_1) - set([intent]))
        if len(neg_1) > 2:
            neg_1 = random.sample(neg_1, 2)

        neg_2 = [s_intent for s_intent in intents if obj in s_intent]
        neg_2 = list(set(neg_2) - set([intent]))

        neg_3 = [s_intent for s_intent in intents if obj not in s_intent and action not in s_intent]
        neg_3 = list(set(neg_3) - set([intent]))
    
        if (len(neg_1) + len(neg_2) + len(neg_3)) > 6:
            neg_3 = random.sample(neg_3, 6 - len(neg_1) - len(neg_2))

        neg = neg_1 + neg_2 + neg_3
        
        # check and add all for testing
        if is_test:
            for neg_intent in intents:
                if neg_intent not in neg and neg_intent != intent:
                    neg.append(neg_intent)

        for n_intent in neg:
            neg_data.append(utterance+"\t"+n_intent+"\t"+str(0))

    return neg_data  

def write_to_file(fname, mylist):
    f = open(fname, "w")
    for line in mylist:
        f.write(line+"\n")
    f.close()

if __name__ == '__main__':
    args = get_parameter()
    path = args.path
    dataset = args.dataset
    numseen = args.numseen

    all_utterances = get_utterances(path+dataset+"/")
    all_utterances = divide_by_intent(all_utterances)

    all_intents = list(set([k.strip() for k,_ in all_utterances.items()]))

    # currently it is set up for SGD 25% ==> 12/46
    seen = random.sample(all_intents,numseen)
    unseen = list(set(all_intents) - set(seen))
    

    train, valid, test = split_data(all_utterances, seen, unseen)


    print("# training utterances, all seen", len(train))
    print("# validation utterances, unseen", len(valid))
    print("# test utterances, unseen", len(test))

    train_data = neg_samples(train, seen, dataset)
    valid_data = neg_samples(valid, unseen, dataset)
    test_data = neg_samples(test, unseen, dataset, is_test=True)

    write_to_file(path+dataset+"/train.txt", train_data)
    write_to_file(path+dataset+"/valid.txt", valid_data)
    write_to_file(path+dataset+"/test.txt", test_data)

    print("dataset saved to", path+dataset)

