import math

# Takes in data and labels, returning a list containing the lines seperated by labels


def format_documents(data, labels):
    documents = [[], []]
    for i, d in enumerate(data):
        documents[int(labels[i])].append(d)
    return documents

# Takes in text corpus, returns all words used


def get_vocab(documents):
    return [word for cls in documents for line in cls for word in line.split(" ")]

# Takes in a given text corpus, and term
# Returns the amount of times the term is used in the corpus


def count_tokens_of_term(document_text_array, term):
    cnt = 0
    for text in document_text_array:
        if text == term:
            cnt += 1
    return cnt

# This function takes in our classes and training documents
# It trains NB via creating a list of conditional probabilities for each word used in the training set
# This list of cond probs includes 2 probabilities, one for each class (for the given word)
# It returns the vocab used in the training set, prior prob for each class and the cond prob list to be used later


def trainMultinomialNB(classes, documents):
    num_classes = len(classes)  # CLEAR
    vocab = get_vocab(documents)
    N = len(documents[0])+len(documents[1])  # CLEAR
    prior = [0 for x in range(num_classes)]  # CLEAR
    condprob = [{}, {}]

    for cls in classes:
        prior[cls] = len(documents[cls])/N
        document_text_array = [word for text in documents[cls]
                               for word in text.split(" ")]
        total_term_count = len(document_text_array)

        for t in vocab:
            term_count = count_tokens_of_term(document_text_array, t)
            prob = (term_count+1)/(total_term_count+len(vocab))
            condprob[cls][t] = prob

    return vocab, prior, condprob

# This function takes in a document/line and tokenizes it
# For each token it grabs the probility of it being in either class
# Creates an overall probability that it belongs in one class or the other
# Then returns the classification corresponding with a higher probability


def applyMultinomialNB(classes, vocab, prior, condprob, document):
    words = [word for word in document.split(' ') if word in vocab]
    score = [0 for cls in classes]
    for cls in classes:
        score[cls] = math.log(prior[cls], 2)
        for word in words:
            score[cls] += math.log(condprob[cls][word], 2)
    return score.index(max(score))

# This function sends off the lines of text to be classified
# It then compares the output to the corresponding label
# Returning the over all accuracy of the classification


def run_naive_bayes(data, labels, classes, vocab, prior, condprob):
    total_cnt = 0
    correct_cnt = 0
    for i, line in enumerate(data):
        if int(labels[i]) == applyMultinomialNB(classes, vocab, prior, condprob, line):
            correct_cnt += 1
        total_cnt += 1
    return (correct_cnt/total_cnt)*100


def read_data(fname):
    with open(fname, "r") as file:
        data = file.read().split("\n")
    return data


train_labels = read_data("trainlabels.txt")
train_data = read_data("traindata.txt")
test_data = read_data("testdata.txt")
test_labels = read_data("testlabels.txt")

# Format training data into two disctint lists by class
train_documents = format_documents(train_data, train_labels)

# Training our NB alg, it returns our vocab, prior prop and cond prob for later use
vocab, prior, condprob = trainMultinomialNB([0, 1], train_documents)

# Test classification with NB, return accuracy based on given labels
test_NB = run_naive_bayes(train_data, train_labels, [
                          0, 1], vocab, prior, condprob)
print(test_NB)
