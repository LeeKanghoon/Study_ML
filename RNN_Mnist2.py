import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False)

# Set Parameter
input_num = 28
hidden_num = 10
output_num = 10

batch_size = 100
epoch = 100

# placeholder
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name = 'x')
y = tf.placeholder(dtype = tf.float32, shape = [None, 10], name = 'y')

# Set Neural Network Parameter
xavier = tf.contrib.layers.xavier_initializer()

w = tf.Variable(xavier(shape = [hidden_num, output_num]))
b = tf.Variable(tf.zeros(shape = [output_num]))

# Forward

cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]

y_pred = tf.nn.softmax(tf.matmul(outputs, w) + b)


# calculate loss
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), axis = 1))

# Backpropagation
optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

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
        batch_x = batch_x.reshape((batch_size, 28, 28))
        gar, loss_temp = sess.run(fetches = [optimizer, loss],  feed_dict = {x: batch_x, y: batch_y})
        loss_e = loss_e + loss_temp / batch_size

    acc = sess.run(accuracy, feed_dict = {x: mnist.test.images.reshape((10000, 28, 28)), y: mnist.test.labels})

    print("Epoch:", '%04d' %(epoch_m + 1), "cost=", "{:.7f}".format(loss_e) ,"accuracy=", "{:.3f}".format(acc))





# end
