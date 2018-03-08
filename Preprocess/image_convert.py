from scipy import misc
import numpy as np
import glob, os
	
def weightedAverage(pixel):
	# #0.2989 * R + 0.5870 * G + 0.1140 * B 
	return 0.2989*pixel[0] + 0.5870*pixel[1] + 0.1140*pixel[2]
	
def greyCodeCnvert(fileName, dest):
	image = misc.imread(fileName)
	grey = np.zeros((image.shape[0], image.shape[1]))
	for rownum in range(len(image)):
		for colnum in range(len(image[rownum])):
			grey[rownum][colnum] = weightedAverage(image[rownum][colnum])	
			
	
	file, ext = os.path.splitext(fileName)
	name = file.split('\\')
	file_name = name[-1]
	saved = dest+file_name+".jpg"
	#print(saved)
	misc.imsave(saved, grey)


source = "D:\\Data\\temp\\*.jpg"	
#source ="D:\Data\Courses\CS674\ISIC-2017_Training_Data_phase3\*.jpg"
print(source)
destination = "D:\\Data\\temp2\\"
count = 0
for infile in glob.glob(source):
	
	#print(infile)
	greyCodeCnvert(infile, destination)
	# count += 1
	# if count == 5:
		# break
	




