from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
total = a + b
print(total)
print(a)
print(b)

c = tf.placeholder()
d = tf.placeholder()
model = tf.add(tf.multiply(c,2),d)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(10):

        session.run(c)
        session.run(d)
        print(session.run(model,feed_dict{c=}))
        print(session.run(c*2+d))


