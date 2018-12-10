import os
import pickle
import glob             # used in getting list of files from directory
import html             # used in clean_text
import random           # used to randomize the dataset
import numpy as np

from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

from collections import Counter     # used in print_dataset_statistics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

############################################################################
# FUNCTIONS Helpers
############################################################################

def get_labels_to_categories_map(y):
    labels = get_class_labels(y)
    return {l: i for i, l in enumerate(labels)}

def get_class_labels(y):
    return numpy.unique(y)

def clean_text(text):
    text = text.rstrip()
    if '""' in text:
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        text = text.replace('\\""', '"')
        text = text.replace('""', '"')
    text = text.replace('\\""', '"')
    text = html.unescape(text)
    text = ' '.join(text.split())
    return text

def parse_file(filename):
    SEPARATOR = "\t"
    data = {}
    for line_id, line in enumerate(open(filename, "r", encoding="utf-8").readlines()):
        try:
            columns = line.rstrip().split(SEPARATOR)
            tweet_id = columns[0]
            sentiment = columns[1]
            text = clean_text(" ".join(columns[2:]))
            if text != "Not Available": data[tweet_id] = (sentiment, text)
        except Exception as e:
            print("\nWrong format in line:{} in file:{}".format(
                line_id, filename))
            raise Exception
    return data

def print_dataset_statistics(y):
    counter = Counter(y)
    print("Total:", len(y))
    statistics = {c: str(counter[c]) + " (%.2f%%)" % (counter[c] / float(len(y)) * 100.0)
                  for c in sorted(counter.keys())}
    print(statistics)

def labels_to_categories(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    y_num = encoder.transform(y)
    return y_num

def prepare_dataset(X, y, pipeline, y_one_hot=True, y_as_is=False):
    try:
        print_dataset_statistics(y)
    except:
        pass
    X = pipeline.fit_transform(X)
    if y_as_is:
        try:
            return X, numpy.asarray(y, dtype=float)
        except:
            return X, y
    # 1 - Labels to categories
    y_cat = labels_to_categories(y)
    if y_one_hot:
        # 2 - Labels to one-hot vectors
        return X, np_utils.to_categorical(y_cat)
    return X, y_cat

def get_class_weights2(y, smooth_factor=0):
    counter = Counter(y)
    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}

############################################################################
# Let's start
############################################################################

WV_CORPUS = "datastories.twitter"
WV_DIM = 300
max_length = 50

wv_filename = "{}.{}d.txt".format(WV_CORPUS, str(WV_DIM))
parsed_filename = "{}.{}d.pickle".format(WV_CORPUS, str(WV_DIM))
# wv_file = os.path.join(os.path.dirname(__file__), wv_filename)
wv_file = os.path.join(os.getcwd(), wv_filename)

# LOAD Word Vector
embeddings_dict = {}

# 1. READ and CREATE from scratch
with open(wv_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = coefs

# 1.a IF we want to save the embeddings_dict TO a file
with open(os.path.join(os.getcwd(), parsed_filename), 'wb') as pickle_file:
    pickle.dump(embeddings_dict, pickle_file)

# 2. LOAD from existing file ?
# with open(_parsed_file, 'rb') as f:
#     vectors = pickle.load(f)

vocab_size = len(embeddings_dict)
print('Found %s word vectors.' % vocab_size)

pos = 0
wv_map = {}
emb_matrix = np.ndarray((vocab_size + 2, WV_DIM), dtype='float32')
for i, (word, vector) in enumerate(embeddings_dict.items()):
    pos = i + 1
    wv_map[word] = pos
    emb_matrix[pos] = vector

pos += 1
wv_map["<unk>"] = pos
emb_matrix[pos] = np.random.uniform(low=-0.05, high=0.05, size=WV_DIM)

# LOAD Dataset
task_folders = os.path.join(os.getcwd(), "Subtask_A/downloaded/")

# Task4Loader(word_indices, text_lengths=max_length, subtask=TASK)
# dataset = SemEvalDataLoader(verbose=False).get_data(task=subtask, years=None, datasets=None, only_semeval=True)
files = glob.glob(task_folders + "*{}.tsv".format('A'))
data = {}
for file in files:
    dataset = parse_file(file)
    data.update(dataset)

dataset = [v for k, v in sorted(data.items())]
random.Random(42).shuffle(dataset)
X = [obs[1] for obs in dataset]     # the texts
y = [obs[0] for obs in dataset]     # the labels
print_dataset_statistics(y)

# training, testing = loader.load_final()
y_one_hot = True
exec(open('CustomPreProcessor.py').read())       # just use what he has done
exec(open('EmbeddingsExtractor.py').read())      # just use what he has done
pipeline = Pipeline([
    ('preprocess', CustomPreProcessor(TextPreProcessor(
        backoff=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
        include_tags={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
        fix_html=True,
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]))),
    ('ext', EmbeddingsExtractor(word_indices=wv_map,
                                max_lengths=max_length,
                                add_tokens=True,
                                unk_policy="random"))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=27)
print("\nPreparing training set...")
training = prepare_dataset(X_train, y_train, pipeline, y_one_hot)
print("\nPreparing validation set...")
validation = prepare_dataset(X_test, y_test, pipeline, y_one_hot)

print("\nPreparing testing set...")         # from the gold data
gold_fname = "SemEval2017-task4-test.subtask-A.english.txt"
gold_folder = os.path.join(os.getcwd(), "Subtask_A/gold/")
gold_parsed = parse_file(gold_folder + gold_fname)
gold_data = [v for k, v in sorted(gold_parsed.items())]
gX = [obs[1] for obs in gold_data]
gy = [obs[0] for obs in gold_data]
gold = prepare_dataset(gX, gy, pipeline, y_one_hot)
testing = gold

# Prepare the Model
exec(open('keras_models.py').read())       # just use what he has done
nn_model = cnn_simple(emb_matrix, max_length)
nn_model = build_attention_RNN(emb_matrix, classes=3, max_length=max_length, layers=2, unit=LSTM,
                               cells=150, bidirectional=True, attention="simple",
                               noise=0.3, clipnorm=1, lr=0.001, loss_l2=0.0001,
                               final_layer=False, dropout_final=0.5, dropout_attention=0.5,
                               dropout_words=0.3, dropout_rnn=0.3, dropout_rnn_U=0.3)

plot_model(nn_model, show_layer_names=True, show_shapes=True, to_file="model_task4_subA.png")

############################################################################
# CALLBACKS
############################################################################
exec(open('callbacks.py').read())       # just use what he has done

metrics = {
    "f1_pn": (lambda y_test, y_pred:
              f1_score(y_test, y_pred, average='macro',
                       labels=[class_to_cat_mapping['positive'],
                               class_to_cat_mapping['negative']])),
    "M_recall": (
        lambda y_test, y_pred: recall_score(y_test, y_pred, average='macro')),
    "M_precision": (
        lambda y_test, y_pred: precision_score(y_test, y_pred,
                                               average='macro'))
}

classes = ['positive', 'negative', 'neutral']
class_to_cat_mapping = get_labels_to_categories_map(classes)
cat_to_class_mapping = {v: k for k, v in
                        get_labels_to_categories_map(classes).items()}

_datasets = {}
_datasets["1-train"] = training,
_datasets["2-val"] = validation
_datasets["3-test"] = testing

metrics_callback = MetricsCallback(datasets=_datasets, metrics=metrics)
weights = WeightsCallback(parameters=["W"], stats=["raster", "mean", "std"])
plotting = PlottingCallback(grid_ranges=(0.5, 0.75), height=5, benchmarks={"SE17": 0.681})

_callbacks = []
_callbacks.append(metrics_callback)
_callbacks.append(plotting)
_callbacks.append(weights)

############################################################################
# END of CALLBACKS
############################################################################

class_weights = get_class_weights2(numpy.asarray(training[1]).argmax(axis=-1), smooth_factor=0)
history = nn_model.fit(training[0], training[1], validation_data=testing, epochs=50, batch_size=50,
                       class_weight=class_weights, callbacks=_callbacks)
pickle.dump(history.history, open("hist_task4_subA.pickle", "wb"))

