from nltk.classify import NaiveBayesClassifier

def word_split(words):
	return dict([(word, True) for word in words.split(" ")])

def get_classifier_accuracy(data, labels, classifier):
	cnt = 0
	for i, d in enumerate(data):
		classification = classifier.classify(word_split(d))
		if classification == labels[i]:
			cnt+=1

	return cnt/len(data)

def read_data(fname):
	with open(fname, "r") as file:
		data = file.read().split("\n")
	return data

train_labels = read_data("trainlabels.txt")
train_data = read_data("traindata.txt")
test_data = read_data("testdata.txt")
test_labels = read_data("testlabels.txt")

training_set = []

for i, data in enumerate(train_data):
	training_set.append((word_split(data), train_labels[i]))

classifier = NaiveBayesClassifier.train(training_set)

train_data_accuracy = 100*get_classifier_accuracy(train_data, train_labels, classifier)
test_data_accuracy = 100*get_classifier_accuracy(test_data, test_labels, classifier)
print("#### Part 1")
print("Ran classifier on training data and training labels, tested on the same")
print("Reported Accuracy: ")
print("{}%".format(train_data_accuracy))

print("#### Part 2")
print("Ran classifier on test data and testing labels, tested on the same")
print("Reported Accuracy: ")
print("{}%".format(test_data_accuracy))




