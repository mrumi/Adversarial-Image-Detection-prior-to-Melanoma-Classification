from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from scipy import misc
import glob
import numpy as np

all_image = []


source = "D:/Data/Courses/CS674/project_part2/cnn/gray/*.jpg"
source_adv = "D:/Data/Courses/CS674/project_part2/cnn/adv/*.jpg"

count = 0
for image_path in glob.glob(source):
	image = misc.imread(image_path)
	all_image.append(image)	
	count += 1
	
images = np.array(all_image)

labels = np.genfromtxt('class.csv', delimiter=',')
print(labels.shape)

all_image_adv = []
for image_path in glob.glob(source_adv):
	image = misc.imread(image_path)
	all_image_adv.append(image)


adv_images = np.array(all_image_adv)

batch_size =50
num_classes = 3
epochs = 50
color = 1

# input image dimensions
img_rows, img_cols = 128, 128

# the data, shuffled and split between train and test sets

ax_train = images[np.r_[0:500]]
bx_train = images[np.r_[1000:1500]]
x_train = np.vstack((ax_train,bx_train))
ax_valid = images[np.r_[500:750]]
bx_valid = images[np.r_[1500:1750]]
x_valid = np.vstack((ax_valid,bx_valid))
x_test = adv_images

print('images shape:', adv_images.shape)

adv_labels = np.genfromtxt('adv_big.csv', delimiter=',')


if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], color, img_rows, img_cols)
	x_valid = x_valid.reshape(x_valid.shape[0], color, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], color, img_rows, img_cols)
	input_shape = (color, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, color)
	x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, color)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, color)
	input_shape = (img_rows, img_cols, color)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('x_train shape:', x_valid.shape)
print('x_train shape:', x_test.shape)

ay_train = labels[np.r_[0:500]]
by_train = labels[np.r_[1000:1500]]
y_train = np.vstack((ay_train,by_train))
ay_valid = labels[np.r_[500:750]]
by_valid = labels[np.r_[1500:1750]]
y_valid = np.vstack((ay_valid,by_valid))
y_test = adv_labels


print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation = 'tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'softmax'))
model.add(Dense(num_classes, activation = 'sigmoid'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
	metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, verbose=0)
prediction = model.predict(x_test)
np.savetxt('adverse.txt', prediction, fmt='%1.4e')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
