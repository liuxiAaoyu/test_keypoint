# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import train_utils

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'cmu_paf_fpn_net', 'The name of the architecture to train.')#t_inception_resnet_v2

tf.app.flags.DEFINE_integer(
    'batch_size', 8, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'train_dir', './log.mse',#'./logInception_resnet/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/xiaoyu/Documents/tfmodels/inception_v4.ckpt',
    'The path to a checkpoint from which to fine-tune.')

#############################################################################
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

# tf.app.flags.DEFINE_string(
#     'train_dir', './log3_/',
#     'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 60,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.00005, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 20,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'CHAI', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')#'train4592'

tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/xiaoyu/Documents/data', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

# tf.app.flags.DEFINE_string(
#     'model_name', 'my_seg_net3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'cmu_paf', 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

# tf.app.flags.DEFINE_integer(
#     'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

# tf.app.flags.DEFINE_string(
#     'checkpoint_path', None,
#     'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,#'InceptionV4/MyLogits/Logits/weights:0,InceptionV4/MyLogits/Logits/biases:0',
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network(
        FLAGS.model_name,
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_cmu_paf_preprocessing(
        preprocessing_name,
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, humans_label, keypoints_label] = provider.get(['image', 'humans', 'keypoints'])

      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      image, gaussian_label, vec_label = image_preprocessing_fn(image, network_fn.default_image_size, 
                                                            network_fn.default_image_size, humans_label, keypoints_label)

      images, gaussian_labels, vec_labels = tf.train.batch(
          [image, gaussian_label, vec_label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
    #   labels = slim.one_hot_encoding(
    #       labels, dataset.num_classes - FLAGS.labels_offset)
    #   labels = tf.squeeze(labels)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, gaussian_labels, vec_labels], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, gaussian_labels, vec_labels = batch_queue.dequeue()
      gaussian_out, vec_out= network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      def cal_loss(gaussian_out, vec_out, gaussian_label, vec_label):
        def eula_loss(x, y, stage, branch):
            name = "stage%d_L%d" % (stage, branch)
            with tf.variable_scope(name):
              return tf.reduce_sum((x-y)*(x-y)) / 2 
        loss = 0
        for i in range(1, 7):
            # lossB1 = eula_loss(gaussian_out[i-1], gaussian_label, i, 1)
            # lossB2 = eula_loss(vec_out[i-1], vec_label, i, 2)
            # tf.losses.add_loss(lossB1)
            # tf.losses.add_loss(lossB2)
            lossB1 = tf.losses.mean_squared_error(gaussian_label, gaussian_out[i-1],reduction=tf.losses.Reduction.SUM)
            lossB2 = tf.losses.mean_squared_error(vec_label, vec_out[i-1],reduction=tf.losses.Reduction.SUM)
            loss +=  lossB1
            loss +=  lossB2
        return loss

      loss = cal_loss(gaussian_out, vec_out, gaussian_labels, vec_labels)
      return loss

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    
    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    #### end_points is the output of clone_fn()
    # end_points = clones[0].outputs
    # for end_point in end_points:
    #   x = end_points[end_point]
    #   summaries.add(tf.summary.histogram('activations/' + end_point, x))
    #   summaries.add(tf.summary.scalar('sparsity/' + end_point,
    #                                   tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    variables_to_train_func = tf.trainable_variables('paf') 
    variables_to_train_base = tf.trainable_variables('InceptionV4')
    
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_losses_base = []
    regularization_losses_func = []
    for i in regularization_losses:
        if i.name[:3] == 'paf':
            regularization_losses_func.append(i)
        else:
            regularization_losses_base.append(i)

    with tf.device(deploy_config.optimizer_device()):
      learning_rate = train_utils._configure_learning_rate(FLAGS, dataset.num_samples, global_step)
      optimizer_base = train_utils._configure_optimizer(FLAGS, learning_rate)
      optimizer_func = train_utils._configure_optimizer(FLAGS, learning_rate*4)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer_base = tf.train.SyncReplicasOptimizer(
          opt=optimizer_base,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
      optimizer_func = tf.train.SyncReplicasOptimizer(
          opt=optimizer_func,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    #variables_to_train = train_utils._get_variables_to_train(FLAGS)

    total_loss_base, clones_gradients_base = model_deploy.optimize_clones(
        clones,
        optimizer_base,
        regularization_losses = regularization_losses_base,
        var_list=variables_to_train_base)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss_base', total_loss_base))

    total_loss_func, clones_gradients_func = model_deploy.optimize_clones(
        clones,
        optimizer_func,
        regularization_losses = regularization_losses_func,
        var_list=variables_to_train_func)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss_func', total_loss_func))

    # Create gradient updates.
    grad_updates_base = optimizer_base.apply_gradients(clones_gradients_base,
                                             global_step=global_step)
    grad_updates_func = optimizer_func.apply_gradients(clones_gradients_func,
                                             global_step=global_step)
    update_ops.append(grad_updates_base)
    update_ops.append(grad_updates_func)

    update_op = tf.group(*update_ops)
    total_loss = tf.add_n([total_loss_base, total_loss_func],name='my_total_loss')
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')


    ###########################
    # Kicks off the training. #
    ###########################
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False,
                            #allow_soft_placement=True,
                            gpu_options=gpu_options)
    saver = tf.train.Saver(max_to_keep=5,
                            keep_checkpoint_every_n_hours=1.0,
                            write_version=2,
                            pad_step_number=False)

    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=train_utils._get_init_fn(FLAGS),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        saver=saver,
        session_config=config,
        sync_optimizer=optimizer_func if FLAGS.sync_replicas else None)


if __name__ == '__main__':
  tf.app.run()
