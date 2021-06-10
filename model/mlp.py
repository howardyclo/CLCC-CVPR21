import tensorflow as tf
slim = tf.contrib.slim

def MLP(x, name, reuse):
    assert x.shape.ndims in [2, 4] # (b, c) or (b, h, w, c)

    if x.shape.ndims == 4:
        x = tf.reduce_mean(x, axis=[1,2]) # (b, c)

    with tf.variable_scope(name, reuse=reuse):
        x = slim.fully_connected(x, 512, tf.nn.relu)
        x = slim.fully_connected(x, 512, tf.nn.relu)
        x = slim.fully_connected(x, 512, tf.nn.relu)
    return x