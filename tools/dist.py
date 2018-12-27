import pybel
import numpy as np
import csv

compound_kinase_affinity = []
with open('unique_binding_KD.csv', 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in csvreader:
			compound_kinase_affinity.append([row[0], row[1], row[2]])

dist_c = []
dist_k = []
skipped = 0

for index in enumerate(compound_kinase_affinity):
    compound = index[1][0]
    kinase = index[1][1].split(',')

    #for j in range(len(kinase)):
    c = next(pybel.readfile('mol', '{}{}.{}'.format('training/', compound, 'mol')))
    try:
    	k = next(pybel.readfile('fasta', '{}{}.{}'.format('training/', kinase[0].strip(), 'fasta')))
    except IOError:
    	#k = next(pybel.readfile(argv.kinase, '{}{}.{}'.format(argv.path, kinase[1].strip(), argv.kinase)))
    	skipped += 1
    	continue

    count = 0
    for a in c:
        if a.atomicnum > 1:
            count += 1

    dist_c.append(count)
    count = 0

    for a in k:
        if a.atomicnum > 1:
            count += 1

    dist_k.append(count)

dist_c = np.array(dist_c)
dist_k = np.array(dist_k)

print("Kinase")
print('Min: ', np.amin(dist_k))
print('Max: ', np.amax(dist_k))
print('Median: ', np.median(dist_k))
print('Mean: ', np.mean(dist_k))
print('Average: ', np.average(dist_k))
print('std: ', np.std(dist_k))
print('Var: ', np.var(dist_k))

print("Compound")
print('Min: ', np.amin(dist_c))
print('Max: ', np.amax(dist_c))
print('Median: ', np.median(dist_c))
print('Mean: ', np.mean(dist_c))
print('Average: ', np.average(dist_c))
print('std: ', np.std(dist_c))
print('Var: ', np.var(dist_c))

print("Skipped: {}".format(skipped))