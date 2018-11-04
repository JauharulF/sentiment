import csv

tw_data, tw_labels = [], []
tw_pos, tw_net, tw_neg = [], [], []
dir_file = "../../twitter-airline-sentiment/"
name_file = "Tweets.csv"
with open(dir_file+name_file, encoding='utf-8') as fin:
    tw_reader = csv.reader(fin, delimiter=',', quotechar='"')
    next(tw_reader, None)   # skip header
    for row in tw_reader:
        # load data (sentence and labels) : positive, neutral, negative
        if row[1] == "neutral":
            tw_net.append(row[10])
        else:
            tw_data.append(row[10])
            if row[1] == "positive":
                tw_pos.append(row[10])
                tw_labels.append(1)
            else:
                tw_neg.append(row[10])
                tw_labels.append(0)

# create dataset: 4000 records, 2000 positives, 2000 negatives
data_sentence, data_labels = [], []
for i in range(2000): # since positive and negative
    data_sentence.append(tw_pos[i])
    data_labels.append(1)
    data_sentence.append(tw_neg[i])
    data_labels.append(0)

num_data = len(data_sentence)

# separate train/dev/test with proportion 8/1/1
num_test = int(0.1 * num_data)
train_data, train_labels, test_data, test_labels = [], [], [], []
train_data = data_sentence[num_test:]
train_labels = data_labels[num_test:]
test_data = data_sentence[:num_test]
test_labels = data_labels[:num_test]

# done with preparation.