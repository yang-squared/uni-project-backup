import tensorflow as tf
import tf_slim as slim


def _log_images(images, name, log_images, samples=10):
    if not log_images:
        return

    width = images[0].get_shape()[0].value
    height = images[0].get_shape()[1].value
    out_channels = images[0].get_shape()[2].value
    imgs = tf.reshape(images[0], [-1, width, height, out_channels])
    imgs = tf.transpose(imgs, [3,1,2,0])
    tf.summary.image(name, imgs, samples)




def network(inputs, num_classes, keep_prob, is_training, log_images):
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu, weights_initializer=tf.contrib.keras.initializers.he_normal()):
        with slim.arg_scope([slim.conv2d], padding="SAME"):
            net = inputs
            _log_images(net, "1.image", log_images)

            net = slim.conv2d(inputs=net, num_outputs=96, kernel_size=[5,5], stride=2, scope='conv1')
            _log_images(net, "2.conv1", log_images)

            net = slim.max_pool2d(inputs=net, kernel_size=[3,3], stride=2, scope='pool1')
            _log_images(net, "3.pool1", log_images)

            net = slim.conv2d(inputs=net, num_outputs=192, kernel_size=[3,3], stride=1, scope='conv2_1')
            _log_images(net, "4.conv2", log_images)

            net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3,3], stride=2, scope='conv2_2')
            _log_images(net, "5.conv2", log_images)

            net = slim.max_pool2d(inputs=net, kernel_size=[2,2], stride=2, scope='pool2')
            _log_images(net, "6.pool2", log_images)
            #net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

            net = slim.flatten(inputs=net, scope='flatten')
            net = slim.fully_connected(inputs=net, num_outputs=128, scope='fc3')
            net = slim.dropout(inputs=net, is_training=is_training, keep_prob=keep_prob, scope='dropout4')
            net = slim.fully_connected(inputs=net, num_outputs=64, scope='fc5')
            net = slim.dropout(inputs=net, is_training=is_training, keep_prob=keep_prob, scope='dropout6')
            net = slim.fully_connected(inputs=net, num_outputs=num_classes, activation_fn=None, scope='fc7')
    return net

            
            

def get_optimizer():
    #tf.train.GradientDescentOptimizer
    return tf.train.AdamOptimizer