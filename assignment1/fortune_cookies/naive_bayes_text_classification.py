import math

def format_documents(data, labels):
	documents = [[],[]]
	for i, d in enumerate(data):
		documents[int(labels[i])].append(d)
	return documents

def get_vocab(documents):
	return [word for cls in documents for line in cls for word in line.split(" ")]

def count_tokens_of_term(document_text_array, term):
	cnt = 0
	for text in document_text_array:
		if text == term:
			cnt+=1
	return cnt

def trainMultinomialNB(classes, documents):
	num_classes = len(classes)
	vocab = get_vocab(documents)
	N = len(documents[0])+len(documents[1])
	prior = [0 for x in range(num_classes)]
	condprob = [{}, {}]

	for cls in classes:
		prior[cls] = len(documents[cls])/N
		document_text_array = [word for text in documents[cls] for word in text.split(" ")]
		total_term_count = len(document_text_array)

		for t in vocab:
			term_count = count_tokens_of_term(document_text_array, t)
			prob = (term_count+1)/(total_term_count+1)
			condprob[cls][t] = prob

	return vocab, prior, condprob

def applyMultinomialNB(classes, vocab, prior, condprob, document):
	words = [word for word in document if word in vocab] 
	score = [0 for cls in classes]
	for cls in classes:
		score[cls] = math.log(prior[cls],2)
		for word in words:
			score[cls] += math.log(condprob[cls][word], 2)
	return score.index(max(score))

def run_naive_bayes(data, labels, classes, vocab, prior, condprob):
	total_cnt = 0
	correct_cnt = 0
	for i, line in enumerate(data):
		if int(labels[i]) == applyMultinomialNB(classes, vocab, prior, condprob, line):
			correct_cnt+=1
		total_cnt+=1
	return (correct_cnt/total_cnt)*100

def read_data(fname):
	with open(fname, "r") as file:
		data = file.read().split("\n")
	return data

train_labels = read_data("trainlabels.txt")
train_data = read_data("traindata.txt")
test_data = read_data("testdata.txt")
test_labels = read_data("testlabels.txt")

documents = format_documents(train_data, train_labels)

vocab, prior, condprob = trainMultinomialNB([0,1], documents)

res = run_naive_bayes(train_data, train_labels, [0,1], vocab, prior, condprob)
print(res)

# print("vocab", vocab)
# print("prior", prior)
# print("condprob", condprob)
