import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

tf.reset_default_graph()
batch_size = 100

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')

num_inp = 784
num_hid = 128
num_lat = 2
learning_rate = 0.0005

xavier = tf.contrib.layers.xavier_initializer()

w1 = tf.Variable(xavier(shape=[num_inp, num_hid]))
w2 = tf.Variable(xavier(shape=[num_hid, num_lat]))
w3 = tf.Variable(xavier(shape=[num_hid, num_lat]))
w4 = tf.Variable(xavier(shape=[num_lat, num_hid]))
w5 = tf.Variable(xavier(shape=[num_hid, num_inp]))

b1 = tf.Variable(tf.zeros(shape=[num_hid]))
b2 = tf.Variable(tf.zeros(shape=[num_lat]))
b3 = tf.Variable(tf.zeros(shape=[num_lat]))
b4 = tf.Variable(tf.zeros(shape=[num_hid]))
b5 = tf.Variable(tf.zeros(shape=[num_inp]))


e1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
e2 = tf.nn.relu(tf.add(tf.matmul(e1, w2), b2))
e3 = tf.nn.relu(tf.add(tf.matmul(e1, w3), b3))

mu = e2
sd = tf.exp(e3)
epsilon = tf.random_normal(shape=tf.shape(sd), mean=0, stddev=1.0)
lat = tf.add(mu, tf.multiply(tf.sqrt(sd), epsilon))

d1 = tf.nn.relu(tf.add(tf.matmul(lat, w4), b4))
y = tf.nn.sigmoid(tf.add(tf.matmul(d1, w5), b5))

img_loss = - tf.reduce_sum(x * tf.log(1e-10+y)+(1-x)*tf.log(1e-10+1-y) , 1)
latent_loss = -0.5 * tf.reduce_sum((1.0 + tf.log(sd) - tf.square(mu) - sd),1)
tot_loss = tf.reduce_sum(tf.add(img_loss, latent_loss))
optimizer = tf.train.AdamOptimizer(0.0005).minimize(tot_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#total_batch = int(np.shape(x)[0] / batch_size)
#print(total_batch)

for epoch in range(1000):
    temp = [0.0, 0.0, 0.0]
    total_batch = int(10000 / batch_size)
    for i in range(total_batch):
        batch = [np.reshape(b, [784]) for b in mnist.test.next_batch(batch_size=batch_size)[0]]
        sess.run(optimizer, feed_dict = {x: batch})
        los1, los2, los3 = sess.run([tot_loss, img_loss, latent_loss], feed_dict = {x: batch})
        temp[0] = temp[0] + (los1 / batch_size)
        temp[1] = temp[1] + (los2 / batch_size)
        temp[2] = temp[2] + (los3 / batch_size)

    print('Epoch:', '%04d' % (epoch + 1), temp[0], sum(temp[1]), sum(temp[2]))



randoms = [np.random.normal(0, 1, 2) for _ in range(20)]
imgs = sess.run(y, feed_dict = {lat: randoms})
imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

for img in imgs:
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.show()
