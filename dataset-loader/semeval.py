dir_file = '../../SemEVAL/2017_English_final/GOLD/Subtask_A/'

train_file = 'twitter-2013train-A.txt'
dev_file   = 'twitter-2013dev-A.txt'
test_file  = 'twitter-2013test-A.txt'

train_13A, dev_13A, test_13A = [], [], []

with open(dir_file+train_file, encoding='utf-8') as fin:
    for row in fin:
        line = row.split('\t')
        data = {'id': line[0], 'label': line[1], 'text': line[2]}
        train_13A.append(data)

with open(dir_file+dev_file, encoding='utf-8') as fin:
    for row in fin:
        line = row.split('\t')
        data = {'id': line[0], 'label': line[1], 'text': line[2]}
        dev_13A.append(data)

with open(dir_file+test_file, encoding='utf-8') as fin:
    for row in fin:
        line = row.split('\t')
        data = {'id': line[0], 'label': line[1], 'text': line[2]}
        test_13A.append(data)

pos, neg, limit = 0, 0, 1000
train_data, train_labels = [], []
for data in train_13A:
    if data['label'] == 'positive' and pos<limit:
        train_data.append(data['text'])
        train_labels.append(1)
        pos += 1
    elif data['label'] == 'negative' and neg<limit:
        train_data.append(data['text'])
        train_labels.append(0)
        neg += 1

pos, neg, limit = 0, 0, 300
dev_data, dev_labels = [], []
for data in dev_13A:
    if data['label'] == 'positive' and pos<limit:
        dev_data.append(data['text'])
        dev_labels.append(1)
        pos += 1
    elif data['label'] == 'negative' and neg<limit:
        dev_data.append(data['text'])
        dev_labels.append(0)
        neg += 1

pos, neg, limit = 0, 0, 500
test_data, test_labels = [], []
for data in test_13A:
    if data['label'] == 'positive' and pos<limit:
        test_data.append(data['text'])
        test_labels.append(1)
        pos += 1
    elif data['label'] == 'negative' and neg<limit:
        test_data.append(data['text'])
        test_labels.append(0)
        neg += 1
