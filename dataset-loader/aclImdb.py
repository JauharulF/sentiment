import os

root_dir = "../../aclImdb/"

word_vocab = {}
with open(root_dir + "imdb.vocab", encoding="utf-8") as fin:
    for row in fin:
        word_vocab[row] = 0

train_pos_list = [fname for fname in os.listdir(root_dir + "train/pos/")]
train_neg_list = [fname for fname in os.listdir(root_dir + "train/neg/")]
test_pos_list  = [fname for fname in os.listdir(root_dir + "test/pos/")] 
test_neg_list  = [fname for fname in os.listdir(root_dir + "test/neg/")] 

train_data, train_labels = [], []
for i in range(len(train_pos_list)):
    with open(root_dir + "train/pos/" + train_pos_list[i], encoding="utf-8") as fin:
        train_data.append(fin.read())
        train_labels.append(1)
    with open(root_dir + "train/neg/" + train_neg_list[i], encoding="utf-8") as fin:
        train_data.append(fin.read())
        train_labels.append(0)

test_data, test_labels = [], []
for i in range(len(test_pos_list)):
    with open(root_dir + "test/pos/" + test_pos_list[i], encoding="utf-8") as fin:
        test_data.append(fin.read())
        test_labels.append(1)
    with open(root_dir + "test/neg/" + test_neg_list[i], encoding="utf-8") as fin:
        test_data.append(fin.read())
        test_labels.append(0)
