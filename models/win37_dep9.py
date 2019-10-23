import tensorflow as tf

slim = tf.contrib.slim

def create_network(inputs, is_training, k, scope="win37_dep9", reuse=False):
	
    num_maps = 64
    kw = k
    kh = k

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=tf.nn.relu, 
            normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
			
            net = slim.conv2d(inputs, 32, [kh, kw], scope='conv_bn_relu1')
            print(net.shape)
            net = slim.conv2d(net, 32, [kh, kw], scope='conv_bn_relu2')
            net = slim.repeat(net, 6, slim.conv2d, num_maps, [kh, kw], scope='conv_bn_relu3_8')

            net = slim.conv2d(net, num_maps, [kh, kw], scope='conv9', activation_fn=None, normalizer_fn=None)
            #net = slim.pool(net, 5, 'MAX', strides=2)
            net = slim.batch_norm(net, is_training=is_training)	

    return net



