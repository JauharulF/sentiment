# load the dataset-loader/semEval loader
exec(open("../dataset-loader/semeval.py").read())

import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import groupby

word_dict = {}
for data in train_13A + dev_13A + test_13A:
    for word in word_tokenize(data['text']):
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1

# try to remove the punctuations token
tobedeleted = []
for word in word_dict:
    if word.startswith('//t.co'): tobedeleted.append(word)
    if word in string.punctuation: tobedeleted.append(word)
    if len([k for k,g in groupby(word)])==1 and word[0] in string.punctuation: tobedeleted.append(word)

for stop in tobedeleted:
    word_dict.pop(stop, None)

# create word_index by sorting the word frequency
word_sorted = sorted(word_dict, key=word_dict.get, reverse=True)
word_index = {v:(k+2) for k,v in enumerate(word_sorted)}
word_index["<PAD>"] = 0
word_index["<UNK>"] = 1  # unknown

# define vocab_size to be used.
vocab_size = 10000
max_length = 50
# load/convert train_data based on word_index
train_data_pad, dev_data_pad, test_data_pad = [], [], []
for data in train_data:
    train_data_pad.append([word_index[word] if word in word_index and word_index[word]<vocab_size else 1 for word in word_tokenize(data)])

for data in dev_data:
    dev_data_pad.append([word_index[word] if word in word_index and word_index[word]<vocab_size else 1 for word in word_tokenize(data)])

for data in test_data:
    test_data_pad.append([word_index[word] if word in word_index and word_index[word]<vocab_size else 1 for word in word_tokenize(data)])

train_data_pad = keras.preprocessing.sequence.pad_sequences(train_data_pad, value=word_index["<PAD>"], padding='post', maxlen=max_length)
dev_data_pad = keras.preprocessing.sequence.pad_sequences(dev_data_pad, value=word_index["<PAD>"], padding='post', maxlen=max_length)
test_data_pad = keras.preprocessing.sequence.pad_sequences(test_data_pad, value=word_index["<PAD>"], padding='post', maxlen=max_length)

# Prepare the Model
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# since the validation model uses separate data provided in dataset, we don't use partial training for validation

history = model.fit(train_data_pad, train_labels, epochs=40, batch_size=40, validation_data=(dev_data_pad, dev_labels), verbose=1)
results = model.evaluate(test_data_pad, test_labels)
print(results)
# [0.6654395518302918, 0.691]

# after training, you may use the model to predict a sentence
# for example, score(model=model, vocab_size=10000, max_length=256, sentence="I love this product.")
def score(model=None, vocab_size=None, max_length=None, sentence=None):
    sent_arr = []
    for word in word_tokenize(sentence.lower()):
        sent_arr.append(word_index[word] if word in word_index and word_index[word]<vocab_size else 2)
    sent_pad = keras.preprocessing.sequence.pad_sequences([sent_arr], value=word_index["<PAD>"], padding='post', maxlen=max_length)
    return model.predict(sent_pad)[0][0]

# cross-check with the training data accuracy
score_trains = [int(round(score(model=model, vocab_size=vocab_size, max_length=max_length, sentence=sentence))) for sentence in train_data]

num_fp = sum([1 if train_labels[k]==0 and score_trains[k]==1 else 0 for k,v in enumerate(train_labels)])
num_tp = sum([1 if train_labels[k]==1 and score_trains[k]==1 else 0 for k,v in enumerate(train_labels)])
num_tn = sum([1 if train_labels[k]==0 and score_trains[k]==0 else 0 for k,v in enumerate(train_labels)])
num_fn = sum([1 if train_labels[k]==1 and score_trains[k]==0 else 0 for k,v in enumerate(train_labels)])

accuracy = (num_tp + num_tn) / (num_tp + num_fp + num_tn + num_fn)
precision = num_tp / (num_tp + num_fp)
recall = num_tp / (num_tp + num_fn)
f1_score = 2 * ((precision * recall) / (precision + recall))

print(num_tp, num_fp, num_tn, num_fn)
print('%.3f %.3f %.3f %.3f' % (accuracy, precision, recall, f1_score))
# 0.847 0.983 0.706 0.822

# measure the performances
score_labels = [int(round(score(model=model, vocab_size=vocab_size, max_length=max_length, sentence=sentence))) for sentence in test_data]

num_fp = sum([1 if test_labels[k]==0 and score_labels[k]==1 else 0 for k,v in enumerate(test_labels)])
num_tp = sum([1 if test_labels[k]==1 and score_labels[k]==1 else 0 for k,v in enumerate(test_labels)])
num_tn = sum([1 if test_labels[k]==0 and score_labels[k]==0 else 0 for k,v in enumerate(test_labels)])
num_fn = sum([1 if test_labels[k]==1 and score_labels[k]==0 else 0 for k,v in enumerate(test_labels)])

precision = num_tp / (num_tp + num_fp)
recall = num_tp / (num_tp + num_fn)
f1_score = 2 * ((precision * recall) / (precision + recall))
accuracy = (num_tp + num_tn) / (num_tp + num_fp + num_tn + num_fn)

print(num_tp, num_fp, num_tn, num_fn)
print('%.3f %.3f %.3f %.3f' % (accuracy, precision, recall, f1_score))
# 0.847 0.983 0.706 0.822

### Create ROC/AOC Graph ###

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_pred_keras = model.predict(test_data_pad).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='positive (area = {:.3f})'.format(auc_keras))
plt.xlabel('False rate')
plt.ylabel('True rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
