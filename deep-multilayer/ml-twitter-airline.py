# load the dataset-loader/twitter-airline loader
exec(open("../dataset-loader/twitter-airline.py").read())

import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from nltk.tokenize import word_tokenize


# load/read the dataset
dataset = tw_data
dataset.extend(tw_net)
word_dict = {}
for sentence in dataset:
    for word in word_tokenize(sentence):
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1

# delete punctuation word
for token in string.punctuation:
    word_dict.pop(token, None)

# delete unnecessary token
word_dict.pop('http', None)
word_dict.pop('https', None)

word_sorted = sorted(word_dict, key=word_dict.get, reverse=True)
word_index = {v:(k+4) for k,v in enumerate(word_sorted)}
# create word_index from dataset, you may use in reversed sort by word frequency
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# define vocab_size to be used.
vocab_size = 10000
# load/convert train_data based on word_index
t_data = []
for sentence in train_data:
    datum = []
    for word in word_tokenize(sentence):
        datum.append(word_index[word] if word in word_index and word_index[word]<vocab_size else 2)
    t_data.append(datum)

train_data = t_data
# load/convert test_data based on word_index
t_data = []
for sentence in test_data:
    datum = []
    for word in word_tokenize(sentence):
        datum.append(word_index[word] if word in word_index and word_index[word]<vocab_size else 2)
    t_data.append(datum)

test_data = t_data

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# define validation set from the train data
validation_size = int(0.1 * len(train_data))
x_val = train_data[:validation_size]
partial_x_train = train_data[validation_size:]
y_val = train_labels[:validation_size]
partial_y_train = train_labels[validation_size:]

# train the model
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data, test_labels)
print(results)

# after training, you may use the model to predict a sentence
# for example, score(model=model, vocab_size=10000, sentence="I love this product.")
def score(model=None, vocab_size=None, sentence=None):
    sent_arr = []
    for word in word_tokenize(sentence.lower()):
        sent_arr.append(word_index[word] if word in word_index and word_index[word]<vocab_size else 2)
    sent_pad = keras.preprocessing.sequence.pad_sequences([sent_arr], value=word_index["<PAD>"], padding='post', maxlen=256)
    return model.predict(sent_pad)[0][0]

score_labels = [int(round(score(model=model, vocab_size=10000, sentence=sentence))) for sentence in data_sentence]

num_fp = sum([1 if data_labels[k]==0 and score_labels[k]==1 else 0 for k,v in enumerate(data_labels)])
num_tp = sum([1 if data_labels[k]==1 and score_labels[k]==1 else 0 for k,v in enumerate(data_labels)])
num_tn = sum([1 if data_labels[k]==0 and score_labels[k]==0 else 0 for k,v in enumerate(data_labels)])
num_fn = sum([1 if data_labels[k]==1 and score_labels[k]==0 else 0 for k,v in enumerate(data_labels)])

accuracy = (num_tp + num_tn) / (num_tp + num_fp + num_tn + num_fn)
precision = num_tp / (num_tp + num_fp)
recall = num_tp / (num_tp + num_fn)
f1_score = 2 * ((precision * recall) / (precision + recall))

print(num_tp, num_fp, num_tn, num_fn)
print(accuracy, precision, recall, f1_score)

### Create ROC/AOC Graph ###

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_pred_keras = model.predict(test_data).ravel()
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
