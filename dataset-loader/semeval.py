import glob
import itertools

dir_file = '../../SemEVAL/2017_English_final/GOLD/Subtask_A/'

# train_file = 'twitter-2013train-A.txt'
# dev_file   = 'twitter-2013dev-A.txt'
# test_file  = 'twitter-2013test-A.txt'

def get_raw_data(filename):
    print(filename)
    dataset = []
    with open(filename, encoding='utf-8') as fin:
        for row in fin:
            line = row.split('\t')
            data = {'id': line[0], 'label': line[1], 'text': line[2]}
            dataset.append(data)
    return dataset

def get_text_data(raw_data, limit):
    pos, neg = 0, 0
    _data, _labels = [], []
    for data in raw_data:
        if data['label'] == 'positive' and pos<limit:
            _data.append(data['text'])
            _labels.append(1)
            pos += 1
        elif data['label'] == 'negative' and neg<limit:
            _data.append(data['text'])
            _labels.append(0)
            neg += 1
    return (_data, _labels)

def get_all_data(raw_data):
    _data, _labels = [], []
    for data in raw_data:
        if data['label'] == 'positive':
            _data.append(data['text'])
            _labels.append(1)
        elif data['label'] == 'negative':
            _data.append(data['text'])
            _labels.append(0)
    return (_data, _labels)


files = glob.glob(dir_file + 'twitter*.txt')
train_raw = list(itertools.chain.from_iterable([get_raw_data(fname) for fname in files]))
test_raw = get_raw_data(dir_file + 'gold/SemEval2017-task4-test.subtask-A.english.txt')

train_data, train_labels = get_text_data(train_raw, limit=7500)
test_data, test_labels = get_text_data(test_raw, limit=2000)
train_data_all, train_labels_all = get_all_data(train_raw)
test_data_all, test_labels_all = get_all_data(test_raw)
