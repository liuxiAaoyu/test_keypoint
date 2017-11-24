


import tensorflow as tf
import matplotlib.pyplot as plt

slim = tf.contrib.slim
reader = tf.TFRecordReader()  
filename_queue = tf.train.string_input_producer(['/home/xiaoyu/Documents/data/CHAIkp_train.tfrecord'])  
_, serialized_example = reader.read(filename_queue)  
features = tf.parse_single_example(serialized_example, features={  
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'humans' : tf.VarLenFeature(dtype=tf.float32),
        'humans/shape' : tf.VarLenFeature(dtype=tf.int64),
        'keypoints' : tf.VarLenFeature(dtype=tf.float32),
        'keypoints/shape' : tf.VarLenFeature(dtype=tf.int64),
    })  
keys_to_features={  
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'humans' : tf.VarLenFeature(dtype=tf.float32),
        'humans/shape' : tf.VarLenFeature(dtype=tf.int64),
        'keypoints' : tf.VarLenFeature(dtype=tf.float32),
        'keypoints/shape' : tf.VarLenFeature(dtype=tf.int64),
    }
items_to_handlers = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'humans': slim.tfexample_decoder.Tensor( 'humans', shape_keys='humans/shape'),
    'keypoints': slim.tfexample_decoder.Tensor('keypoints', shape_keys='keypoints/shape'),
}

decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                   items_to_handlers)
[tf_image, tf_humans, tf_keypoints] = decoder.decode(serialized_example,
                                        ['image', 'humans', 'keypoints'])

import preprocessing.cmu_paf_preprocessing
import numpy as np

distorted_image, gaussian, vec = preprocessing.cmu_paf_preprocessing.preprocess_for_train2(tf_image, 368, 368, tf_humans, tf_keypoints, None)

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(1,1000):
        a,b,c = sess.run([distorted_image, gaussian, vec])
        a = a/2 + 0.5
        print(a.shape)
        print(c)
        print(b)
        plt.imshow(a)
        plt.show()
        ttt = np.zeros(c[:,:,0].shape)
        for i in range(14):
            ttt = ttt + c[:,:,i*2]
        plt.imshow(ttt)
        plt.show()
        ttt = np.zeros(b[:,:,0].shape)
        for i in range(14):
            ttt = ttt + b[:,:,i]
        plt.imshow(ttt)
        plt.show()
    coord.request_stop()
coord.join(threads) 