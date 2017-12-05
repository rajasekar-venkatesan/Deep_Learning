import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#HELPER FUNCTIONS

#INIT WEIGHTS
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

#INIT BIAS
def init_bias(shape):
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)

#CONV2D
def conv2d(x,W):
    # x --> [batch, H, W, Channels]
    # W --> [filter_H, filter_W, Channels_IN, Channels_OUT]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#POOLING
def max_pool_2by2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#CONVOLUTION LAYER
def convolution_layer(input_x,shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)

#FULLY CONNECTED LAYER
def normal_full_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size,size])
    b = init_bias([size])
    return tf.matmul(input_layer,W)+b

#PLACEHOLDERS
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])

#LAYERS
x_image = tf.reshape(x,[-1,28,28,1])
convo1 = convolution_layer(x_image,shape=[5,5,1,32])
convo1_pool = max_pool_2by2(convo1)

convo2 = convolution_layer(convo1_pool,shape=[5,5,32,64])
convo2_pool = max_pool_2by2(convo2)

convo2_flat = tf.reshape(convo2_pool,[-1,7*7*64])
full_layer1 = tf.nn.relu(normal_full_layer(convo2_flat,1024))

#DROPOUT
hold_prob = tf.placeholder(tf.float32)
full1_dropout = tf.nn.dropout(full_layer1,keep_prob = hold_prob)

y_pred = normal_full_layer(full1_dropout,10)

#LOSS
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits=y_pred))

#OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

steps = 1000

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        if i%100 == 0:
            print('ON STEP: {}'.format(i))
            print('ACCURACY:')
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
            print('\n')