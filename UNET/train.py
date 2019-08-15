# coding:utf-8

import time
import os
import pandas as pd
import tensorflow as tf


def image_augmentation(image, mask):
    """Returns (maybe) augmented images

    (1) Random flip (left <--> right)
    (2) Random flip (up <--> down)
    (3) Random brightness
    (4) Random hue

    Args:
        image (3-D Tensor): Image tensor of (H, W, C)
        mask (3-D Tensor): Mask image tensor of (H, W, 1)

    Returns:
        image: Maybe augmented image (same shape as input `image`)
        mask: Maybe augmented mask (same shape as input `mask`)
    """
    concat_image = tf.concat([image, mask], axis=-1)

    maybe_flipped = tf.image.random_flip_left_right(concat_image)
    maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)

    image = maybe_flipped[:, :, :-1]
    mask = maybe_flipped[:, :, -1:]

    # image = tf.image.random_brightness(image, 0.7)
    # image = tf.image.random_hue(image, 0.1)

    return image, mask


def get_image_mask(queue, image_size, augmentation=True):
    """Returns `image` and `mask`

    Input pipeline:
        Queue -> CSV -> FileRead -> Decode JPEG

    (1) Queue contains a CSV filename
    (2) Text Reader opens the CSV
        CSV file contains two columns
        ["path/to/image.jpg", "path/to/mask.jpg"]
    (3) File Reader opens both files
    (4) Decode JPEG to tensors

    Notes:
        height, width = 640, 960

    Returns
        image (3-D Tensor): (640, 960, 3)
        mask (3-D Tensor): (640, 960, 1)
    """
    #
    text_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = text_reader.read(queue)

    image_path, mask_path = tf.decode_csv(csv_content, record_defaults=[[""], [""]])

    image_file = tf.read_file(image_path)
    mask_file = tf.read_file(mask_path)

    image = tf.image.decode_jpeg(image_file, channels=3)
    image.set_shape([image_size, image_size, 3])
    image = tf.cast(image, tf.float32)

    mask = tf.image.decode_jpeg(mask_file, channels=1)
    mask.set_shape([image_size, image_size, 1])
    mask = tf.cast(mask, tf.float32)
    mask = mask / (tf.reduce_max(mask) + 1e-7)

    if augmentation:
        image, mask = image_augmentation(image, mask)

    return image, mask


def conv_conv_pool(input_, n_filters, training, name, pool=True, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
                                   # , kernel_regularizer=regularizer)
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upsample_concat(inputA, input_B, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    upsample = upsampling_2D(inputA, size=(2, 2), name=name)

    return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))


def upsampling_2D(tensor, name, size=(2, 2)):
    """Upsample/Rescale `tensor` by size

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))

    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    H, W, _ = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name="upsample_{}".format(name))


def make_unet(X, training, channel_size):
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    # net = X / 127.5 - 1
    # net = tf.layers.conv2d(net, 3, (1, 1), name="color_space_adjust")
    mean, variance = tf.nn.moments(X, [1, 2], keep_dims=True)
    # net = tf.div(tf.subtract(X, mean), tf.sqrt(variance))
    net = tf.subtract(X, mean)
    # net = tf.layers.batch_normalization(X, training=True, name="bn_0")
    conv1, pool1 = conv_conv_pool(net, [channel_size, channel_size], training, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [channel_size*2, channel_size*2], training, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [channel_size*4, channel_size*4], training, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [channel_size*8, channel_size*8], training, name=4)
    conv5 = conv_conv_pool(pool4, [channel_size*16, channel_size*16], training, name=5, pool=False)

    up6 = upsample_concat(conv5, conv4, name=6)
    conv6 = conv_conv_pool(up6, [channel_size*8, channel_size*8], training, name=6, pool=False)

    up7 = upsample_concat(conv6, conv3, name=7)
    conv7 = conv_conv_pool(up7, [channel_size*4, channel_size*4], training, name=7, pool=False)

    up8 = upsample_concat(conv7, conv2, name=8)
    conv8 = conv_conv_pool(up8, [channel_size*2, channel_size*2], training, name=8, pool=False)

    up9 = upsample_concat(conv8, conv1, name=9)
    conv9 = conv_conv_pool(up9, [channel_size, channel_size], training, name=9, pool=False)

    # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    return tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=None, padding='same')
                            # ,kernel_regularizer=regularizer)


def focal_loss_op(y_pred, y_true, alpha=0.5, gamma=2):
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    sigmoid_p = tf.nn.sigmoid(pred_flat)

    per_entry_cross_ent = - alpha * true_flat * (tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0) ** gamma) * \
                          tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (1.0 - true_flat) * (tf.clip_by_value(sigmoid_p, 1e-8, 1.0) ** gamma) * \
                          tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    return tf.reduce_mean(per_entry_cross_ent)

def loss_op(y_pred, y_true):
    """Returns a (approx) IOU score

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU scoreta
    """
    # H, W, _ = y_pred.get_shape().as_list()[1:]
    #
    # pred_flat = tf.reshape(y_pred, [-1, H * W])
    # true_flat = tf.reshape(y_true, [-1, H * W])

    # loss_sigmoid = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    pos_weight = 5
    loss_sigmoid = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pos_weight=pos_weight, name=None))

    return loss_sigmoid


def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


def make_train_op(y_pred, y_true):
    """Returns a training operation

    Loss function = - IOU(y_pred, y_true)

    IOU is

        (the area of intersection)
        --------------------------
        (the area of two boxes)

    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)

    Returns:
        train_op: minimize operation
    """
    # loss = -IOU_(y_pred, y_true)
           # + tf.losses.get_regularization_loss()
    loss = loss_op(y_pred, y_true)

    global_step = tf.train.get_or_create_global_step()

    optim = tf.train.AdamOptimizer(learning_rate=0.00001)
    return optim.minimize(loss, global_step=global_step)


#python train.py --epochs 300 --batch-size 4 --gpu 0 --logdir ./loghp2 --ckdir ./modelhp


def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",
                        default=50,
                        type=int,
                        help="Number of epochs (default: 1)")

    parser.add_argument("--batch-size",
                        default=10,
                        type=int,
                        help="Batch size (default: 4)")

    parser.add_argument("--image-size",
                        default=768,
                        type=int,
                        help="image size (default: 256)")

    parser.add_argument("--gpu",
                        default=0,
                        type=int,
                        help="gpu")

    parser.add_argument("--logdir",
                        default="logs",
                        help="Tensorboard log directory (default: logdir)")

    parser.add_argument("--ckdir",
                        default="models",
                        help="Checkpoint directory (default: models)")

    flags = parser.parse_args()
    return flags


def main(flags):
    # 后缀名，供多次测试时使用
    _POST = '_dem'
    train = pd.read_csv("./train{:s}.csv".format(_POST), header=None)
    n_train = train.shape[0]

    test = pd.read_csv("./test{:s}.csv".format(_POST), header=None)
    n_test = test.shape[0]

    current_time = time.strftime("%m_%d_%H_%M_%S")
    train_logdir = os.path.join(flags.logdir, "train", current_time)
    test_logdir = os.path.join(flags.logdir, "test", current_time)
    if not os.path.exists(flags.ckdir):
        os.mkdir(flags.ckdir)
    ckdir = os.path.join(flags.ckdir, current_time)
    print("train log dir: %s" % train_logdir)
    print("test log dir: %s" % test_logdir)
    print("models dir: %s" % ckdir)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 3], name="X")
    y = tf.placeholder(tf.float32, shape=[None, flags.image_size, flags.image_size, 1], name="y")
    mode = tf.placeholder(tf.bool, name="mode")

    pred = make_unet(X, mode, channel_size=8)
    # pred = make_hed(X, mode, channel_size=8)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = make_train_op(pred, y)

    # IOU_op = IOU_(pred, y)
    # IOU_op = tf.Print(IOU_op, [IOU_op])
    # tf.summary.scalar("IOU", IOU_op)

    loss = loss_op(pred, y)
    tf.summary.scalar("loss", loss)

    pred = tf.nn.sigmoid(pred, name='sigmoid')

    pred_label = tf.cast(tf.greater(pred, 0.5), dtype=tf.float32)
    H, W = pred_label.get_shape().as_list()[1:3]

    pred_flat = tf.reshape(pred_label, [-1, H * W])
    true_flat = tf.reshape(y, [-1, H * W])

    intersection = pred_flat * true_flat
    pre = tf.reduce_sum(intersection) / (tf.reduce_sum(pred_flat) + 1e-7)
    rec = tf.reduce_sum(intersection) / (tf.reduce_sum(true_flat) + 1e-7)
    f1_score = 2 / (1 / pre + 1 / rec + 1e-7)
    tf.summary.scalar("precision", pre)
    tf.summary.scalar("rec", rec)
    tf.summary.scalar("f1_score", f1_score)

    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)

    tf.summary.image("image", X)
    tf.summary.image("label", y*255)
    tf.summary.image("Predicted Mask", pred*255)

    tf.summary.histogram("Predicted Mask", pred)

    train_csv = tf.train.string_input_producer(['train{:s}.csv'.format(_POST)])
    test_csv = tf.train.string_input_producer(['test{:s}.csv'.format(_POST)])
    train_image, train_mask = get_image_mask(train_csv, image_size=flags.image_size)
    test_image, test_mask = get_image_mask(test_csv, image_size=flags.image_size, augmentation=False)

    X_batch_op, y_batch_op = tf.train.shuffle_batch([train_image, train_mask],
                                                    batch_size=flags.batch_size,
                                                    capacity=flags.batch_size * 3,
                                                    min_after_dequeue=flags.batch_size * 2,
                                                    allow_smaller_final_batch=False)

    X_test_op, y_test_op = tf.train.batch([test_image, test_mask],
                                          batch_size=flags.batch_size,
                                          capacity=flags.batch_size * 2,
                                          allow_smaller_final_batch=False)

    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_summary_writer = tf.summary.FileWriter(test_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=10)

        if os.path.exists(ckdir) and tf.train.checkpoint_exists(ckdir):
            latest_check_point = tf.train.latest_checkpoint(ckdir)
            saver.restore(sess, latest_check_point)
        else:
            os.mkdir(ckdir)

        try:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(flags.epochs):
                print("------------------------------------------")
                print("epoch %d" % epoch)
                n_train_num = n_train//flags.batch_size
                for step in range(n_train_num):

                    X_batch, y_batch = sess.run([X_batch_op, y_batch_op])

                    _, loss_value, step_summary, pre_value, rec_value = sess.run(
                        [train_op, loss, summary_op, pre, rec],
                         feed_dict={X: X_batch,
                                   y: y_batch,
                                   mode: True})

                    print("step:{:d}, train precision:{:.4f}, recall:{:.4f}, loss:{:.4f}".format(step, pre_value, rec_value, loss_value))

                    if step % 10 == 0:
                        train_summary_writer.add_summary(step_summary, epoch*n_train_num + step)

                n_test_num = n_test // flags.batch_size
                for step in range(n_test_num):

                    X_test, y_test = sess.run([X_test_op, y_test_op])

                    loss_value, step_summary, pre_value, rec_value = sess.run(
                        [loss, summary_op, pre, rec],
                        feed_dict={X: X_test,
                                   y: y_test,
                                   mode: False})

                    print("step:{:d}, test precision:{:.4f}, recall:{:.4f}, loss:{:.4f}".format(step, pre_value, rec_value, loss_value))

                    if step % 10 == 0:
                        test_summary_writer.add_summary(step_summary, epoch*n_train_num + int(step*n_train_num/n_test_num))
                if epoch % 2 == 0:
                    saver.save(sess, "{:s}/model{:d}.ckpt".format(ckdir, epoch))
                    print('save model %d' % epoch)
        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(ckdir))


if __name__ == '__main__':
    flags = read_flags()
    print(flags)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(flags.gpu)
    main(flags)
