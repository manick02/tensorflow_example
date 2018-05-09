# this is a get to started of linear regression
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf

num_points = 1000
vectors_set = []
W = 0.1
b = 0.4
for i in range(num_points):
    x1 = np.random.normal(0.0, 1.0)
    # adding noise
    nd = np.random.normal(0.0, 0.05)
    y1 = W * x1 + b
    y1 = y1+nd
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
#print('x-data')
#print(x_data)
#print(y_data)


# plot.plot(x_data, y_data, 'r*', label='Original data')
# plot.legend()
# plot.show()
# show = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b = tf.Variable(tf.zeros([1]))
# y = W * x_data + b
#x_data defining loss function


with tf.name_scope("LinearRegression") as scope:
   W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="Weights")
   b = tf.Variable(tf.zeros([1]))
   y = W * x_data + b
print('y')
print(y)
with tf.name_scope("LossFunction") as scope:
   loss = tf.reduce_mean(tf.square(y - y_data))
print('loss')
print(loss)
optimizer = tf.train.GradientDescentOptimizer(0.6)
train = optimizer.minimize(loss)
loss_summary = tf.summary.scalar("loss", loss)
w_ = tf.summary.histogram("W", W)
b_ = tf.summary.histogram("b", b)
merged_op = tf.summary.merge_all()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
writer_tensorboard = tf.summary.FileWriter('/tmp/tensorflow_logs', sess.graph)
for i in range(16):
   sess.run(train)
   print(i, sess.run(W), sess.run(b), sess.run(loss))
   plot.plot(x_data, y_data, 'ro', label='Original data')
   plot.plot(x_data, sess.run(W)*x_data + sess.run(b))
   plot.xlabel('X')
   plot.xlim(-2, 2)
   plot.ylim(0.1, 0.6)
   plot.ylabel('Y')
   plot.legend()
   plot.show()

sess.close()
