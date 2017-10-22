from __future__ import print_function

from collections import defaultdict
try:
	import cPickle as pickle
except ImportError:
	import pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.datasets import mnist, cifar10
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

from scipy import misc
import glob


np.random.seed(1337)

K.set_image_data_format('channels_first')


def build_generator(latent_size):	
	cnn = Sequential()
	cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
	cnn.add(Dense(128 * 8 * 8, activation='relu'))
	cnn.add(Reshape((128, 8, 8)))	
	cnn.add(UpSampling2D(size=(4, 4)))
	cnn.add(Conv2D(256, 5, padding='same', activation='relu', kernel_initializer='glorot_normal'))	
	cnn.add(UpSampling2D(size=(4, 4)))
	cnn.add(Conv2D(128, 5, padding='same',activation='relu',kernel_initializer='glorot_normal'))	
	cnn.add(Conv2D(1, 2, padding='same', activation='tanh', kernel_initializer='glorot_normal'))	
	latent = Input(shape=(latent_size, ))	
	image_class = Input(shape=(1,), dtype='int32')	
	cls = Flatten()(Embedding(3, latent_size, embeddings_initializer='glorot_normal')(image_class))	
	h = layers.multiply([latent, cls])
	fake_image = cnn(h)
	return Model([latent, image_class], fake_image)


def build_discriminator(color, img_rows, img_cols):	
	cnn = Sequential()
	cnn.add(Conv2D(32, 3, padding='same', strides=2, input_shape=(color, img_rows, img_cols)))
	cnn.add(LeakyReLU())
	cnn.add(Dropout(0.3))
	cnn.add(Conv2D(64, 3, padding='same', strides=2))
	cnn.add(LeakyReLU())
	cnn.add(Dropout(0.3))
	cnn.add(Conv2D(128, 3, padding='same', strides=2))
	cnn.add(LeakyReLU())
	cnn.add(Dropout(0.3))
	cnn.add(Conv2D(256, 3, padding='same', strides=1))
	cnn.add(LeakyReLU())
	cnn.add(Dropout(0.3))
	cnn.add(Flatten())
	image = Input(shape=(color, img_rows, img_cols))

	features = cnn(image)
	
	fake = Dense(1, activation='sigmoid', name='generation')(features)
	aux = Dense(3, activation='softmax', name='auxiliary')(features)

	return Model(image, [fake, aux])

if __name__ == '__main__':
	
	color = 1
	row = 128
	col = 128	
	
	all_image = []
		
	source = "C:/Users/vslabsal/Desktop/674/gray/*.jpg"	
	count = 0
	for image_path in glob.glob(source):
		image = misc.imread(image_path)
		all_image.append(image)
		count = count+1
		if count == 750:
			break
		#print(image.shape)
		

	images = np.array(all_image)	
	X_train = images[np.r_[0:500]]
	X_test = images[np.r_[500:750]]

	labels = np.genfromtxt('labels.csv', delimiter=',')
	y_train = labels[np.r_[0:500]]
	y_test = labels[np.r_[500:750]]				
	
	X_train = (X_train.astype(np.float32) - 127.5) / 127.5
	X_train = np.expand_dims(X_train, axis=1)

	X_test = (X_test.astype(np.float32) - 127.5) / 127.5
	X_test = np.expand_dims(X_test, axis=1)	
	
	num_train, num_test = X_train.shape[0], X_test.shape[0]
	print(num_train,num_test)
		
		
	epochs = 100
	batch_size = 50
	latent_size = 50
	
	adam_lr = 0.0002
	adam_beta_1 = 0.5
	

	# build the discriminator
	discriminator = build_discriminator(color, row, col)
	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
		loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

	# build the generator
	generator = build_generator(latent_size)
	generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss='binary_crossentropy')

	latent = Input(shape=(latent_size, ))
	image_class = Input(shape=(1,), dtype='int32')

	# get a fake image
	fake = generator([latent, image_class])

	# we only want to be able to train generation for the combined model
	discriminator.trainable = False
	fake, aux = discriminator(fake)
	combined = Model([latent, image_class], [fake, aux])

	combined.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), 
		loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

	train_history = defaultdict(list)
	test_history = defaultdict(list)
	
	for epoch in range(epochs):
		print('Epoch {} of {}'.format(epoch + 1, epochs))

		num_batches = int(X_train.shape[0] / batch_size)
		progress_bar = Progbar(target=num_batches)

		epoch_gen_loss = []
		epoch_disc_loss = []

		for index in range(num_batches):
			progress_bar.update(index)
			# generate a new batch of noise
			noise = np.random.uniform(-1, 1, (batch_size, latent_size))

			# get a batch of real images
			image_batch = X_train[index * batch_size:(index + 1) * batch_size]
			label_batch = y_train[index * batch_size:(index + 1) * batch_size]						
			sampled_labels = np.random.randint(0, 3, batch_size)
			
			generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)			
			X = np.concatenate((image_batch, generated_images))
			y = np.array([1] * batch_size + [0] * batch_size)
			
			aux_y = np.concatenate((label_batch, sampled_labels), axis=0)
			
			epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))
			
			noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
			sampled_labels = np.random.randint(0, 3, 2 * batch_size)

			trick = np.ones(2 * batch_size)

			epoch_gen_loss.append(combined.train_on_batch([noise, 
				sampled_labels.reshape((-1, 1))],[trick, sampled_labels]))

		print('\nTesting for epoch {}:'.format(epoch + 1))
		
		noise = np.random.uniform(-1, 1, (num_test, latent_size))		
		# sample some labels from p_c and generate images from them
		sampled_labels = np.random.randint(0, 3, num_test)
		
		generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=False)		
		
		X = np.concatenate((X_test, generated_images))
		y = np.array([1] * num_test + [0] * num_test)

		aux_y = np.concatenate((y_test, sampled_labels), axis=0)
		# see if the discriminator can figure itself out...
		discriminator_test_loss = discriminator.evaluate(X, [y, aux_y], verbose=False)
		discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

		# make new noise
		noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
		sampled_labels = np.random.randint(0, 3, 2 * num_test)
		
		
		trick = np.ones(2 * num_test)

		generator_test_loss = combined.evaluate([noise, 
			sampled_labels.reshape((-1, 1))],[trick, sampled_labels], verbose=False)

		generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
		
		train_history['generator'].append(generator_train_loss)
		train_history['discriminator'].append(discriminator_train_loss)

		test_history['generator'].append(generator_test_loss)
		test_history['discriminator'].append(discriminator_test_loss)

		print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format('component', *discriminator.metrics_names))
		print('-' * 65)

		ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
		print(ROW_FMT.format('generator (train)', *train_history['generator'][-1]))
		print(ROW_FMT.format('generator (test)', *test_history['generator'][-1]))
		print(ROW_FMT.format('discriminator (train)',*train_history['discriminator'][-1]))
		print(ROW_FMT.format('discriminator (test)',*test_history['discriminator'][-1]))
		
		
		noise = np.random.uniform(-1, 1, (1, latent_size))
		sampled_labels = np.random.randint(0, 3, 1).reshape(-1, 1)
		
		generated_images = generator.predict([noise, sampled_labels], verbose=0)
		print("passed")
		
		img = (np.concatenate([r.reshape(-1, 128)
			for r in np.split(generated_images, 1)], axis=-1) * 127.5 + 127.5).astype(np.uint8)

		Image.fromarray(img).save('plot_epoch_{0:03d}_generated.png'.format(epoch))

	
