import tensorflow as tf
from nets import inception_v4
from nets import inception_v2
from nets import inception_utils
import numpy as np

slim = tf.contrib.slim

def my_arg_scpoe(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001):
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc
    
def load_inception_seg(sess, inception_path):
    tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.TRAINING],'/home/xiaoyu/Documents/tfmodels/inceptionv4')
    inputs = sess.graph.get_tensor_by_name('input:0')
    l4o = sess.graph.get_tensor_by_name('InceptionV4/Mixed_4a/concat:0')
    l5o = sess.graph.get_tensor_by_name('InceptionV4/Mixed_5e/concat:0')
    l6o = sess.graph.get_tensor_by_name('InceptionV4/Mixed_6h/concat:0')
    l7o = sess.graph.get_tensor_by_name('InceptionV4/Mixed_7d/concat:0')
    return inputs, l4o, l5o, l6o, l7o

def my_seg_net4(inputs, num_classes = None, is_training = None, scope = None):
    end_points={}
    #last_layer, _ = load_inception(input_image, num_classes)
    with tf.variable_scope(scope, 'my_seg_net3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
            net = inputs

            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv1_2_2')
            end_points['block1'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1_2')
            # Block 2.
            #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 32, [1, 1], scope='conv2_1_1x1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
            end_points['block2'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
            # Block 3.
            #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 64, [1, 1], scope='conv3_1_1x1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
            net = slim.conv2d(net, 64, [1, 1], scope='conv3_2_1x1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv3_3')
            end_points['block3'] = net
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
            # Block 4.
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_1')
            net = slim.conv2d(net, 128, [1, 1], scope='conv4_1_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_2')
            net = slim.conv2d(net, 128, [1, 1], scope='conv4_2_1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv4_3')
            end_points['block4'] = net

            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')
            # Block 5.
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
            net = slim.conv2d(net, 256, [1, 1], scope='conv5_1_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
            net = slim.conv2d(net, 256, [1, 1], scope='conv5_2_1x1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
            end_points['block5'] = net

            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool5')
            #Block 6
            net = slim.conv2d(net, 1024, [3, 3], scope='conv6_1')
            net = slim.conv2d(net, 512, [1, 1], scope='conv6_1_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv6_2')
            net = slim.conv2d(net, 512, [1, 1], scope='conv6_2_1x1')
            net = slim.conv2d(net, 1024, [3, 3], scope='conv6_3')
            end_points['block6'] = net

            return net, end_points

my_seg_net4.default_image_size = 448
my_seg_net4.default_image_height = 640#1280/2
my_seg_net4.default_image_width = 960#1920/2
my_seg_net4.my_seg_net1_arg_scope = my_arg_scpoe

def stage1_block(net, num_p, branch):
    net = slim.conv2d(net, 128, [3, 3], scope="conv1_stage%d_L%d" % (1, branch))
    net = slim.conv2d(net, 128, [3, 3], scope="conv1_stage%d_L%d" % (1, branch))
    net = slim.conv2d(net, 128, [3, 3], scope="conv1_stage%d_L%d" % (1, branch))
    net = slim.conv2d(net, 512, [1, 1], scope="conv1_stage%d_L%d" % (1, branch))
    net = slim.conv2d(net, num_p, [1, 1], scope="conv1_stage%d_L%d" % (1, branch))

def stageT_block(net, num_p, stage, branch):
    net = slim.conv2d(net, 128, [7, 7], scope="conv1_stage%d_L%d" % (stage, branch))
    net = slim.conv2d(net, 128, [7, 7], scope="conv1_stage%d_L%d" % (stage, branch))
    net = slim.conv2d(net, 128, [7, 7], scope="conv1_stage%d_L%d" % (stage, branch))
    net = slim.conv2d(net, 128, [7, 7], scope="conv1_stage%d_L%d" % (stage, branch))
    net = slim.conv2d(net, 128, [7, 7], scope="conv1_stage%d_L%d" % (stage, branch))
    net = slim.conv2d(net, 512, [1, 1], scope="conv1_stage%d_L%d" % (stage, branch))
    net = slim.conv2d(net, num_p, [1, 1], scope="conv1_stage%d_L%d" % (stage, branch))

def paf_net(inputs,):
    net, end_points = inception_v4.inception_v4_base(inputs)
    net = end_points['Mixed_5e']
    gaussian_out = []
    vec_out = []
    with tf.variable_scope(scope, 'paf'):
        stage1_branch1_out = stage1_block(net, 14, 1)
        stage1_branch2_out = stage1_block(net, 28, 2)
        net = tf.concat([net, stage1_branch1_out, stage1_branch2_out], axis = -1)
        gaussian_out.append(stage1_branch1_out)
        vec_out.append(stage1_branch2_out)
        for sn in range(2, 7):
            stageT_branch1_out = stageT_block(net, 14, sn)
            stageT_branch2_out = stageT_block(net, 28, sn)
            net = tf.concat([net, stageT_branch1_out, stageT_branch2_out], axis = -1)
            gaussian_out.append(stageT_branch1_out)
            vec_out.append(stageT_branch2_out)
    return gaussian_out, vec_out
paf_net.default_image_size = 299
paf_net.default_image_height =299#640#448#1280
paf_net.default_image_width = 299#960#448#1920
paf_net.arg_scope = my_arg_scpoe

def cal_loss(gaussian_out, vec_out, gaussian_label, vec_label):
    def eula_loss(x, y, stage, branch):
        name = "stage%d_L%d" % (stage, branch)
        with tf.variable_scope(scope, name):
          return tf.reduce_sum((x-y)*(x-y) / 2)
    loss = 0
    for i in range(1, 7):
        loss += eula_loss(gaussian_out[i], gaussian_label, i, 1)
        loss += eula_loss(gaussian_out[i], gaussian_label, i, 2)
    with tf.variable_scope(scope, 'total'):
        tf.add_to_collection('LOSSES', loss)
    return loss
    
