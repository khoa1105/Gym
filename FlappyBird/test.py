import random

experiences = []
experiences.append([[1,2,3], 199, 0, [1,2,3,4]])
experiences.append([[1,2,3,4,5], 199, 0, [1,2,3,4,5,6]])
experiences.append([[1], 199, 0, [1,2]])
print(experiences)
removes = random.sample(experiences,2)
for rm in removes:
	experiences.remove(rm)
print(experiences)