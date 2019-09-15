import math

pclass_yrate = {
	"1st": {"rate": 203/325, "total": 325},
	"2nd": {"rate": 118/285, "total": 285},
	"3rd": {"rate": 178/706, "total": 706},
	"crew": {"rate": 212/885, "total": 885}
}

Age_yrate = {
	"Adult": {"rate": 654/2092, "total": 2092},
	"Child": {"rate": 57/109, "total": 109},
}


Sex_yrate = {
	"Male": {"rate": 367/1731, "total": 1731},
	"Female": {"rate": 344/470, "total": 470}
}

Male_age_yrate = {
	"adult": {"rate": 338/1667, "total": 1667},
	"child": {"rate": 29/64, "total": 64}
}

Male_class_yrate = {
	"1st": {"rate": 62/180, "total": 180},
	"2nd": {"rate": 25/179, "total": 179},
	"3rd": {"rate": 88/510, "total": 510},
	"crew": {"rate": 192/862, "total": 862}
}

total_passengers = 2202
total_males = 1731
total_females = 470

def get_attribute_entropy(entropies, total_records, yrates):
	entropy = 0
	for ent in entropies:
		entropy += (yrates[ent]["total"]/total_records)*entropies[ent]
	return entropy

def entropy(yrate):
	return -yrate*math.log(yrate) - (1-yrate)*math.log(1-yrate)

def get_value_entropies(yrate):
	entropies = {}
	for value in yrate:
		entropies[value] = entropy(yrate[value]["rate"])
	return entropies

def print_entropy_data(yrate, total_records, entropy_name):
	print()
	print("***", entropy_name,"***")
	entropy = get_value_entropies(yrate)
	print(entropy)
	print( "total", entropy_name, "entropy", get_attribute_entropy(entropy, total_records, yrate) )
	print()

def print_root_data():
	# class
	print_entropy_data(pclass_yrate, total_passengers, "class")

	# age
	print_entropy_data(Age_yrate, total_passengers, "age")

	# sex
	print_entropy_data(Sex_yrate, total_passengers, "sex")

def print_male_data():
	# male-age
	print_entropy_data(Male_age_yrate, total_males, "male-age")

	# male-class
	print_entropy_data(Male_class_yrate, total_males, "male-class")
	

# print_root_data()
print_male_data()








