import numpy as np
import tensorflow as tf
from PIL import Image

from scipy import misc
import glob
from numpy import array

train_image = []
train_source = "D:/Data/Courses/CS674/project_part2/train_net/*.jpg"

for image_path in glob.glob(train_source):
	image = misc.imread(image_path)
	train_image.append(image)

images = np.array(train_image)

test_image = []
test_source = "D:/Data/Courses/CS674/project_part2/adv_mix/*.jpg"
for image_path in glob.glob(test_source):
	image = misc.imread(image_path)
	test_image.append(image)

test_images = np.array(test_image)

xx_train = images[np.r_[750:1500]]
xx_valid = images[np.r_[1500:1750]]
xx_test = test_images

tr = []
for im in xx_train:
	xt = im.flatten()
	tr.append(xt)
x_train = np.array(tr)

vd = []
for im in xx_valid:
	v = im.flatten()
	vd.append(v)
x_valid = np.array(vd)

ts = []
for im in xx_test:
	yt = im.flatten()
	ts.append(yt)
x_test = np.array(ts)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')

train_labels = np.genfromtxt('labels.csv', delimiter=',')
train_labels = train_labels.astype('int64')

test_labels = np.genfromtxt('adv_labels.csv', delimiter=',')
test_labels = test_labels.astype('int64')

y_train = train_labels[np.r_[750:1500]]
y_valid = train_labels[np.r_[1500:1750]]
y_test = test_labels

n_labels = 3
n_channels = 1
image_width = 28
n_hidden = 256
n_input = image_width ** 2
bottleneck = 10

graph = tf.Graph()
with graph.as_default():	
	x = tf.placeholder(tf.float32, [None, n_input])	
	y = tf.placeholder(tf.int64, [None])
    
	W = {
		'1': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_input, n_hidden]), 0)),
		'2': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
		'out': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_labels]), 0)),
	}

	b = {
		'1': tf.Variable(tf.zeros([n_hidden])),
		'2': tf.Variable(tf.zeros([n_hidden])),
		'out': tf.Variable(tf.zeros([n_labels])),
	}

	def gelu_fast(__x):
		return 0.5 * __x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (__x + 0.044715 * tf.pow(__x, 3))))
	f = gelu_fast

	def model(_x):		
		h1 = f(tf.matmul(_x, W['1']) + b['1'])
		h2 = f(tf.matmul(h1, W['2']) + b['2'])
		out = tf.matmul(h2, W['out']) + b['out']

		return out

	pred = model(x)

	starter_learning_rate = 0.001
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=y))
	lr = tf.constant(0.001)
	optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

	wrong_pred = tf.not_equal(tf.argmax(pred, 1), y)
	compute_error = 100. * tf.reduce_mean(tf.to_float(wrong_pred))

sess = tf.InteractiveSession(graph=graph)
tf.initialize_all_variables().run()

batch_size = 25
training_epochs = 30

num_batches = int(x_train.shape[0] / batch_size)
ce_ema = 2.3            
err_ema = 0.9
risk_loss_ema = 0.3     
learning_rate = 0.001
num_batch_valid = int(x_valid.shape[0] / batch_size)
for epoch in range(training_epochs):
	if epoch >= 20:
		learning_rate = 0.0001
	start = 0
	for i in range(num_batches):		
		bx = x_train[np.r_[start:start+batch_size]]
		by = y_train[np.r_[start:start+batch_size]]
		start = start + batch_size
		_, err, l = sess.run([optimizer, compute_error, loss], feed_dict={x: bx, y: by, lr: learning_rate})
		ce_ema = ce_ema * 0.95 + 0.05 * l
		err_ema = err_ema * 0.95 + 0.05 * err    
	start = 0
	for i in range(num_batch_valid):		
		bx = x_valid[np.r_[start:start+batch_size]]
		by = y_valid[np.r_[start:start+batch_size]]
		start = start + batch_size		
		_, err, l = sess.run([optimizer, compute_error, loss], feed_dict={x: bx, y: by, lr: learning_rate})
		ce_ema = ce_ema * 0.95 + 0.05 * l
		err_ema = err_ema * 0.95 + 0.05 * err

	print('Epoch number:', epoch, 'Error EMA:', err_ema, 'Loss EMA', ce_ema)
print('Done training')


l1_distances = []
l2_distances = []
linf_distances = []



start = 0
adversarial_index = []
clean_index = []

for i in range(x_test.shape[0]):
	image = x_test[np.r_[i:i+1]]
	true_y = y_test[np.r_[i:i+1]]
	start = start + 1    
	#print(i)
	confidence = sess.run(tf.nn.softmax(model(image))[0, true_y])

	if confidence < 0.5:
		fooled = 'not_fooled'        
		adversarial_index.append(start)
	else:
		fooled = 'fooled'
		clean_index.append(start)
        

tp=0
fp=0
tn=0
fn=0

for i in adversarial_index :
	if i< 49 or i> 159:
		tp+=1
	else:
		fp+=1

		
for i in clean_index :
	if i< 49 or i> 159:
		fn+=1
	else:
		tn+=1

print(tp)
print(tn)
print(fp)
print(fn)

