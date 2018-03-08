from PIL import Image
import glob, os
from resizeimage import resizeimage

dest = "D:\\Data\\temp\\"
source ="D:\Data\Courses\CS674\ISIC-2017_Training_Data_phase3\*.jpg"


size = 128, 128

count = 0

for infile in glob.glob(source):
	file, ext = os.path.splitext(infile)
	name = file.split('\\')
	file_name = name[-1]	
	im = Image.open(infile)
	im.thumbnail(size,Image.ANTIALIAS)
	#im.resize((size))
	
	
	cover = resizeimage.resize_cover(im, [128, 128])
	saved = dest+file_name+".jpg"
	cover.save(saved, im.format)
	
	
	#im.save()
	count += 1
	if count == 100:
		break