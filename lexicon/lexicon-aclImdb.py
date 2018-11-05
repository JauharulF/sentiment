# load the dataset-loader/aclImdb loader
exec(open("../dataset-loader/aclImdb.py").read())

# load the lexicon/lexiconSA class
exec(open("./lexiconSA.py").read())

senti_file = "../../myexp/SentiWordNet.txt"
lsa = LexiconSA(filename = senti_file)

score_labels = [0 if lsa.score(sentence)<0 else 1 for sentence in test_data]

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
