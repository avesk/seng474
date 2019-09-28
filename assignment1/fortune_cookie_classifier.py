from nltk.classify import NaiveBayesClassifier

with open("trianlabels.txt") as file:
	train_labels = file.read()