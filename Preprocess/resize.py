from PIL import Image
import glob, os


all_image = []

source = "D:\\Data\\Courses\\CS674\\project_part2\\cnn\\adv\\*.jpg"
dest = "D:\\Data\\Courses\\CS674\\project_part2\\adv_mix\\"


for image_path in glob.glob(source):  	
	image = Image.open(image_path) 
	file, ext = os.path.splitext(image_path)
	name = file.split('\\')
	file_name = name[-1]		
	img = image.resize((int(28),int(28)), Image.ANTIALIAS)
	name = dest+file_name+ '.jpg'
	img.save(name)    
    

