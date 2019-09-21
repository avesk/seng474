''' 

Description: This script calculates probabilities using the Naive Bayes Method based on data obtained from
the titanic text file

Authors: Avery Kushner, Derek Siemens 

'''

import math

second_child_male = {
	"2nd": 118.0/285.0,
	"Child": 57.0/109,
	"Male": 367.0/1731.0
}

second_adult_female = {
	"2nd": 118.0/285.0,
	"Adult": 654.0/2092.0,
	"Female": 344.0/470.0
}

y_total_prob = 711.0/2201.0

def naive_bayes(probabilities, pyes):
	y_probability_multiples = 1
	n_probability_multiples = 1

	for key in probabilities.keys():
		y_probability_multiples*= probabilities[key]
		n_probability_multiples*= (1 - probabilities[key])

	y_probability_multiples*=pyes
	n_probability_multiples*=(1-pyes)
	alpha = 1/(y_probability_multiples + n_probability_multiples)

	return (y_probability_multiples*alpha*100, n_probability_multiples*alpha*100)


def print_result():
	print("#### Naive Bayes ####")
	print("2nd child male ?")
	pyes, pno = naive_bayes(second_child_male, y_total_prob)
	print("Probability of yes: {}%".format(pyes))
	print("Probability of no: {}%".format(pno))

	print()

	print("2nd adult female ?")
	pyes, pno = naive_bayes(second_adult_female, y_total_prob)
	print("Probability of yes: {}%".format(pyes))
	print("Probability of no: {}%".format(pno))

print_result()

