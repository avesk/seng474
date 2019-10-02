''' 

Description: This script calculates probabilities using the Naive Bayes Method based on data obtained from
the titanic text file

Authors: Avery Kushner, Derek Siemens 

'''

import math

second_child_male_yes = {
	"2nd": 118.0/711.0,
	"Child": 57.0/711.0,
	"Male": 367.0/711.0
}
second_child_male_no = {
	"2nd": 167.0/1490.0,
	"Child": 52.0/1490.0,
	"Male": 1364.0/1490.0
}
second_adult_female_yes = {
	"2nd": 118.0/711.0,
	"Adult": 654.0/711.0,
	"Female": 344.0/711.0
}
second_adult_female_no = {
	"2nd": 167.0/1490.0,
	"Adult": 1438.0/1490.0,
	"Female": 126.0/1490.0
}

y_total_prob = 711.0/2201.0

def naive_bayes(cond_prob_yes, cond_prob_no, pyes):
	y_probability_multiples = 1
	n_probability_multiples = 1

	for key in cond_prob_yes.keys():
		y_probability_multiples *= cond_prob_yes[key]
	for key in cond_prob_no.keys():
		n_probability_multiples *= cond_prob_no[key]

	y_probability_multiples*=pyes
	n_probability_multiples*=(1-pyes)
	print("P(survived | E ) alpha*{}".format(y_probability_multiples))
	print("P(didnt survive | E ) alpha*{}".format(n_probability_multiples))
	alpha = 1/(y_probability_multiples + n_probability_multiples)
	print("Our Alpha Value is 1/P(survived| E) + P(didnt survive | E) ={}".format(alpha))

	return (y_probability_multiples*alpha*100, n_probability_multiples*alpha*100)


def print_result():
	print("#### Naive Bayes ####")
	print("2nd child male ?")
	pyes, pno = naive_bayes(second_child_male_yes, second_child_male_no, y_total_prob)
	print("Probability of yes: {}%".format(pyes))
	print("Probability of no: {}%".format(pno))
	print('Thus classified as No')

	print()

	print("2nd adult female ?")
	pyes, pno = naive_bayes(second_adult_female_yes, second_adult_female_no, y_total_prob)
	print("Probability of yes: {}%".format(pyes))
	print("Probability of no: {}%".format(pno))
	print('Thus classified as Yes')

print_result()

