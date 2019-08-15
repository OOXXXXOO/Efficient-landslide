import tensorflow as tf


def conv_conv_pool(input_, n_filters, training, name, pool=True, activation=tf.nn.relu):

    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def make_hed(net, training, channel_size):

    mean, variance = tf.nn.moments(net, [1, 2], keep_dims=True)
    net = tf.subtract(net, mean)
    shape = net.get_shape().as_list()[1:3]

    conv1, pool1 = conv_conv_pool(net, [channel_size, channel_size], training, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [channel_size*2, channel_size*2], training, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [channel_size*4, channel_size*4, channel_size*4], training, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [channel_size*8, channel_size*8, channel_size*8], training, name=4)
    conv5 = conv_conv_pool(pool4, [channel_size*8, channel_size*8, channel_size*8], training, name=5, pool=False)

    b5_conv = tf.layers.conv2d(conv5, 1, (1, 1), activation=None, padding='same', name="b5_conv")
    b5 = tf.image.resize_bilinear(b5_conv, size=shape, align_corners=True, name="b5")

    b4_conv = tf.layers.conv2d(conv4, 1, (1, 1), activation=None, padding='same', name="b4_conv")
    b4 = tf.image.resize_bilinear(b4_conv, size=shape, align_corners=True, name="b4")

    b3_conv = tf.layers.conv2d(conv3, 1, (1, 1), activation=None, padding='same', name="b3_conv")
    b3 = tf.image.resize_bilinear(b3_conv, size=shape, align_corners=True, name="b3")

    b2_conv = tf.layers.conv2d(conv2, 1, (1, 1), activation=None, padding='same', name="b2_conv")
    b2 = tf.image.resize_bilinear(b2_conv, size=shape, align_corners=True, name="b2")

    b1_conv = tf.layers.conv2d(conv1, 1, (1, 1), activation=None, padding='same', name="b1_conv")
    b1 = tf.image.resize_bilinear(b1_conv, size=shape, align_corners=True, name="b1")

    concat = tf.concat(axis=-1, values=[b1, b2, b3, b4, b5], name="concat")

    conv6 = conv_conv_pool(concat, [5, 5], training, name=6, pool=False)
    final = tf.layers.conv2d(conv6, 1, (1, 1), activation=None, padding='same', name="final")

    return final

if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=(None, 768, 768, 3), name='data_placeholder')
    make_hed(x, True, 16)
