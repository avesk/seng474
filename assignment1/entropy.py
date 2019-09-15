import math

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

pclass_yrate = {
	"1st": {"rate": 203/325, "total": 325},
	"2nd": {"rate": 118/285, "total": 285}, 
	"3rd": {"rate": 178/706, "total": 706},
	"crew": {"rate": 212/885, "total": 885}
}

plass_entropies = {}
for pclass in pclass_yrate:
	plass_entropies[pclass] = entropy(pclass_yrate[pclass]["rate"])

# print(plass_entropies)
# print( "toal entropy", get_attribute_entropy(plass_entropies, 2201, pclass_yrate) )


Age_yrate = {
	"Adult": {"rate": 654/2092, "total": 2092},
	"Child": {"rate": 57/109, "total": 109},
}


Sex_yrate = {
	"Male": {"rate": 367/1731, "total": 1731},
	"Female": {"rate": 344/470, "total": 470}
}

sex_entropies = get_value_entropies(Sex_yrate)
print(sex_entropies)
print( "total entropy", get_attribute_entropy(sex_entropies, 2201, Sex_yrate) )


