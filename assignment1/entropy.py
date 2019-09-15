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

pclass_entropies = get_value_entropies(pclass_yrate)
print(plass_entropies)
print( "total class entropy", get_attribute_entropy(plass_entropies, 2201, pclass_yrate) )

age_entropies = get_value_entropies(Age_yrate)
print(Age_yrate)
print( "total age entropy", get_attribute_entropy(age_entropies, 2201, Age_yrate) )

sex_entropies = get_value_entropies(Sex_yrate)
print(sex_entropies)
print( "total sex entropy", get_attribute_entropy(sex_entropies, 2201, Sex_yrate) )
