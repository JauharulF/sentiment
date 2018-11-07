# load the dataset-loader/aclImdb loader
exec(open("../dataset-loader/aclImdb.py").read())

import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from nltk.tokenize import sent_tokenize, word_tokenize

# load/read the dataset
all_data = train_data[:]
all_data.extend(test_data)

# option 1: create own dictionary
# word_dict = {}
# for paragraph in all_data:
#     for sentence in sent_tokenize(paragraph):
#         for word in word_tokenize(sentence):
#             if word in word_dict:
#                 word_dict[word] += 1
#             else:
#                 word_dict[word] = 1

# delete punctuation word
# for token in string.punctuation:
#     word_dict.pop(token, None)

# delete unnecessary token
# word_dict.pop('http', None)
# word_dict.pop('https', None)

# option 2: use the vocabulary given by the dataset
for paragraph in all_data:
    for sentence in sent_tokenize(paragraph):
        for word in word_tokenize(sentence.lower()):
            if word in word_vocab:
                word_vocab[word] += 1

len(word_vocab)
# word_sorted = sorted(word_dict, key=word_dict.get, reverse=True) # option 1
word_sorted = sorted(word_vocab, key=word_vocab.get, reverse=True) # option 2
word_index = {v:(k+4) for k,v in enumerate(word_sorted)}
# create word_index from dataset, you may use in reversed sort by word frequency
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# define vocab_size to be used.
vocab_size = 10000
# load/convert train_data based on word_index
train_data_pad = []
for paragraph in train_data:
    datum = []
    for sentence in sent_tokenize(paragraph):
        for word in word_tokenize(sentence.lower()):
            if word in string.punctuation:
                continue
            datum.append(word_index[word] if word in word_index and word_index[word]<vocab_size else 2)
    train_data_pad.append(datum)

# load/convert test_data based on word_index
test_data_pad = []
for paragraph in test_data:
    datum = []
    for sentence in sent_tokenize(paragraph):
        for word in word_tokenize(sentence.lower()):
            if word in string.punctuation:
                continue
            datum.append(word_index[word] if word in word_index and word_index[word]<vocab_size else 2)
    test_data_pad.append(datum)

train_data_pad = keras.preprocessing.sequence.pad_sequences(train_data_pad, value=word_index["<PAD>"], padding='post', maxlen=256)
test_data_pad = keras.preprocessing.sequence.pad_sequences(test_data_pad, value=word_index["<PAD>"], padding='post', maxlen=256)
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# define validation set from the train data
validation_size = int(0.4 * len(train_data_pad))
x_val = train_data_pad[:validation_size]
partial_x_train = train_data_pad[validation_size:]
y_val = train_labels[:validation_size]
partial_y_train = train_labels[validation_size:]

# train the model
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data_pad, test_labels)
print(results)

# after training, you may use the model to predict a sentence
# for example, score(model=model, vocab_size=10000, sentence="I love this product.")
def score(model=None, vocab_size=None, sentence=None):
    sent_arr = []
    for word in word_tokenize(sentence.lower()):
        sent_arr.append(word_index[word] if word in word_index and word_index[word]<vocab_size else 2)
    sent_pad = keras.preprocessing.sequence.pad_sequences([sent_arr], value=word_index["<PAD>"], padding='post', maxlen=256)
    return model.predict(sent_pad)[0][0]

score_labels = [int(round(score(model=model, vocab_size=10000, sentence=sentence))) for sentence in test_data]

num_fp = sum([1 if test_labels[k]==0 and score_labels[k]==1 else 0 for k,v in enumerate(test_labels)])
num_tp = sum([1 if test_labels[k]==1 and score_labels[k]==1 else 0 for k,v in enumerate(test_labels)])
num_tn = sum([1 if test_labels[k]==0 and score_labels[k]==0 else 0 for k,v in enumerate(test_labels)])
num_fn = sum([1 if test_labels[k]==1 and score_labels[k]==0 else 0 for k,v in enumerate(test_labels)])

accuracy = (num_tp + num_tn) / (num_tp + num_fp + num_tn + num_fn)
precision = num_tp / (num_tp + num_fp)
recall = num_tp / (num_tp + num_fn)
f1_score = 2 * ((precision * recall) / (precision + recall))

print(num_tp, num_fp, num_tn, num_fn)
print(accuracy, precision, recall, f1_score)

# cross-check with the training data accuracy
score_trains = [int(round(score(model=model, vocab_size=10000, sentence=sentence))) for sentence in train_data]

num_fp = sum([1 if train_labels[k]==0 and score_trains[k]==1 else 0 for k,v in enumerate(train_labels)])
num_tp = sum([1 if train_labels[k]==1 and score_trains[k]==1 else 0 for k,v in enumerate(train_labels)])
num_tn = sum([1 if train_labels[k]==0 and score_trains[k]==0 else 0 for k,v in enumerate(train_labels)])
num_fn = sum([1 if train_labels[k]==1 and score_trains[k]==0 else 0 for k,v in enumerate(train_labels)])

accuracy = (num_tp + num_tn) / (num_tp + num_fp + num_tn + num_fn)
precision = num_tp / (num_tp + num_fp)
recall = num_tp / (num_tp + num_fn)
f1_score = 2 * ((precision * recall) / (precision + recall))

print(num_tp, num_fp, num_tn, num_fn)
print(accuracy, precision, recall, f1_score)

### Create ROC/AOC Graph ###

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_pred_keras = keras_model.predict(test_data_pad).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.4)
plt.ylim(0.6, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
