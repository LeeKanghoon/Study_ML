import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# placeholder
x = tf.placeholder(dtype = tf.float32, shape = [None, 784], name = 'x')
y = tf.placeholder(dtype = tf.float32, shape = [None, 10], name = 'y')

# Set Parameter
input_num = 784
hidden_num = 100
output_num = 10

batch_size = 100
epoch = 20

# Set Neural Network Parameter
xavier = tf.contrib.layers.xavier_initializer()

w1 = tf.Variable(xavier(shape = [input_num, hidden_num]))
w2 = tf.Variable(xavier(shape = [hidden_num, output_num]))

b1 = tf.Variable(tf.zeros(shape = [hidden_num]))
b2 = tf.Variable(tf.zeros(shape = [output_num]))

# Forward
h1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))
y_pred = tf.nn.softmax(tf.add(tf.matmul(h1, w2), b2))

# calculate loss
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), axis = 1))

# Backpropagation
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# Accuracy
is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Tensorflow Setting
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# mini_batch optimize
for epoch_m in range(epoch):

    total_batch = int(mnist.train.images.shape[0] / batch_size)
    loss_e = 0.0
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        gar, loss_temp = sess.run(fetches = [optimizer, loss],  feed_dict = {x: batch_x, y: batch_y})
        loss_e = loss_e + loss_temp / batch_size

    acc = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels})

    print("Epoch:", '%04d' %(epoch_m + 1), "cost=", "{:.7f}".format(loss_e) , "accuracy=", "{:.3f}".format(acc))
