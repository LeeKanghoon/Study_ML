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
x = tf.placeholder(dtype = tf.float32, shape = [100, 28, 28, 1], name = 'x')
x_split = tf.split(x, 28, axis=1)
y = tf.placeholder(dtype = tf.float32, shape = [None, 10], name = 'y')

x_t = tf.placeholder(dtype = tf.float32, shape = [10000, 28, 28, 1], name = 'x_t')
x_t_split = tf.split(x_t, 28, axis=1)
y_t = tf.placeholder(dtype = tf.float32, shape = [10000, 10], name = 'y_t')

# Set Neural Network Parameter
xavier = tf.contrib.layers.xavier_initializer()

w1 = tf.Variable(xavier(shape = [input_num, hidden_num]))
w2 = tf.Variable(xavier(shape = [hidden_num, hidden_num]))
w3 = tf.Variable(xavier(shape = [hidden_num, output_num]))

b1 = tf.Variable(tf.zeros(shape = [hidden_num]))
#b2 = tf.Variable(tf.zeros(shape = [hidden_num]))
b3 = tf.Variable(tf.zeros(shape = [output_num]))

# Forward

s = {}
s[-1] = np.zeros(shape = (batch_size, hidden_num), dtype = 'f')

for i in range(len(x_split)):
    x_temp = tf.reshape(x_split[i], [batch_size, 28])
    cal1 = tf.add(tf.matmul(x_temp, w1), b1)
    cal2 = tf.matmul(s[i-1], w2)
    s[i] = tf.nn.tanh(tf.add(cal1, cal2))

y_pred = tf.nn.softmax(tf.add(tf.matmul(s[28-1],w3), b3))

# calculate loss
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), axis = 1))

# Backpropagation
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Accuracy
is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Test_Accuracy

s_t = {}
s_t[-1] = np.zeros(shape = (10000, hidden_num), dtype = 'f')

for i in range(len(x_t_split)):
    x_t_temp = tf.reshape(x_t_split[i], [10000, 28])
    cal1_t = tf.add(tf.matmul(x_t_temp, w1), b1)
    cal2_t = tf.matmul(s_t[i-1], w2)
    s_t[i] = tf.nn.tanh(tf.add(cal1_t, cal2_t))

y_t_pred = tf.nn.softmax(tf.add(tf.matmul(s_t[28-1],w3), b3))

is_correct_t = tf.equal(tf.argmax(y_t_pred, 1), tf.argmax(y_t, 1))
accuracy_t = tf.reduce_mean(tf.cast(is_correct_t, tf.float32))

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

    acc = sess.run(accuracy_t, feed_dict = {x_t: mnist.test.images, y_t: mnist.test.labels})

    print("Epoch:", '%04d' %(epoch_m + 1), "cost=", "{:.7f}".format(loss_e) ,"accuracy=", "{:.3f}".format(acc))





# end
