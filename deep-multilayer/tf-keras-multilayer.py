import numpy as np
import tensorflow as tf
from tensorflow import keras

# load/read the dataset
# create word_index from dataset, you may use in reversed sort by word frequency
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# load train data
# load test data

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# define validation set from the train data
validation_size = 0.1 * len(train_data)
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
from nltk.tokenize import word_tokenize

def score(model=None, vocab_size=None, sentence=None):
    sent_arr = []
    for word in word_tokenize(sentence.lower()):
        sent_arr.append(word_index[word] if word in word_index and word_index[word]<vocab_size else 2)
    sent_pad = keras.preprocessing.sequence.pad_sequences([sent_arr], value=word_index["<PAD>"], padding='post', maxlen=256)
    return model.predict(sent_pad)[0][0]
