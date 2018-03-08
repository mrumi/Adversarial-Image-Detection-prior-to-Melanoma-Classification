import csv
import numpy as np

ifile = open('adv.csv', "r")
reader = csv.reader(ifile,delimiter=',')

truth = []

count = 0
c1=0
c2=0
c3=0
for row in reader:
	
	# convert numbers to numpy float64
	r1 = np.float64(row[0])
	r2 = np.float64(row[1])
	r3 = np.float64(row[2])		
		
	label = -1
	if (r1 == 1.0):		
		truth.append(0)	
	elif (r2 == 1.0):
		truth.append(1)		
	elif (r3 == 1.0):
		truth.append(2)		
	
ifile.close()
labels = np.array(truth)
np.savetxt("adv_labels.csv", labels, delimiter=",")
