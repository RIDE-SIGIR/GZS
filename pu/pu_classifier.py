import argparse

from sklearn.svm import SVC
from pulearn import ElkanotoPuClassifier
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


parser = argparse.ArgumentParser(description='Positive Unlabelled Classifier')
parser.add_argument('-path', action="store", default="../data/")
parser.add_argument('-dataset', action="store", default="dstc8-sgd")
args = parser.parse_args()

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
embed = hub.Module(module_url)

#positive
path = args.path
dataset = args.dataset
pos_intents = open(path+dataset+"/pu_pos.txt", encoding='utf-8', errors='ignore').read().split('\n')
pos_intents = [l for l in pos_intents if len(l) > 0]

#unlabelled
unlabel_intents = open(path+dataset+"/pu_unlabelled.txt", encoding='utf-8', errors='ignore').read().split('\n')
unlabel_intents = [l for l in unlabel_intents if len(l) > 0]

# positive label for training
y1 = np.zeros(len(pos_intents))
y1.fill(1)

#no label for dev
y2 = np.zeros(len(unlabel_intents))
final_y = np.concatenate((y1, y2), axis=None)


all_intents = pos_intents + unlabel_intents

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    final_vectors = session.run(embed(all_intents))


### model
print("Traing PU classifier")
svc = SVC(C=10, kernel='rbf', probability=True)
pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.1)
pu_estimator.fit(final_vectors, final_y)


#test
seen = open(path+dataset+"/pu_test_pos.txt", encoding='utf-8', errors='ignore').read().split('\n')
unseen = open(path+dataset+"/pu_test_neg.txt", encoding='utf-8', errors='ignore').read().split('\n')

t1 = np.zeros(len(seen))
t1.fill(1)
t2 = np.zeros(len(unseen))
t2.fill(-1)
test_y = np.concatenate((t1, t2), axis=None)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    test_vectors = session.run(embed(unlabel_intents))
 

y_pred = pu_estimator.predict(test_vectors)

correct = 0
for index in range(y_pred.shape[0]):
    if y_pred[index] == test_y[index]:
        correct+=1

result = precision_recall_fscore_support(test_y, y_pred, average='weighted')

print(correct/y_pred.shape[0] * 100)
print(result)
