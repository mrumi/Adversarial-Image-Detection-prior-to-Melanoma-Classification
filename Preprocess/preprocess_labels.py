import csv
import numpy as np

ifile = open('ISIC-2017_Training_Part3_GroundTruth.csv', "r")
reader = csv.reader(ifile,delimiter=',')

truth = []
rownum = 0
i=0
for row in reader:
	i+=1
	l = []
	if rownum ==0:
		rownum += 1
		continue
	
	# convert numbers to numpy float64
	r1 = np.float64(row[1])
	r2 = np.float64(row[2])	
	
	
	
	if (r1 == 1.0) or (r2 == 1.0):
		r3 = np.float64(0)
	#elif (r1 == 0.0) and (r2 == 0.0):
	else:
		r3 = np.float64(1)
		
	l.append(r1)
	l.append(r2)
	l.append(r3)
	#print(l)
		
	#print(row)
	#print(l)
	c = l.count(1)
	if c > 1:
		break
	truth.append(l)
	
	
ifile.close()
labels = np.array(truth)
np.savetxt("class.csv", labels, delimiter=",")


