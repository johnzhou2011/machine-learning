from __future__ import print_function
import tensorflow as tf
import numpy as np

# data creation
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.3 + 0.3

# tensorflow
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# session
sess = tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
