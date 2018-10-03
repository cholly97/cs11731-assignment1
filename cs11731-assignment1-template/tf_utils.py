import tensorflow as tf

def linear(input_, output_size, scope=None, initializer = None, with_w=False, reuse = False):
    shape = input_.get_shape().as_list()
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, initializer = initializer)
    bias = tf.get_variable("bias", [output_size], initializer = initializer)
    return tf.matmul(input_, matrix) + bias