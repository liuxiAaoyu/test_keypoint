import tensorflow as tf
import matplotlib.pyplot as plt

#a=tf.Variable(tf.truncated_normal([None,1280,1918,3])) 
# a = tf.placeholder(dtype=tf.float32,shape=[4,1280,1920,3])
# logits,endpoints = my_seg_net.my_seg_net1(a,2)
# print(endpoints)

#from datasets import dataset_factory

slim = tf.contrib.slim
reader = tf.TFRecordReader()  
filename_queue = tf.train.string_input_producer(['/home/xiaoyu/Documents/data/ai_challenger_keypoint_validation_20170911/train.tfrecord'])  
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
                                        
# decoder = slim.tfexample_decoder.Image( 'image/encoded', 'image/format')
# image = decoder.tensors_to_item(features)
# decoder2 = slim.tfexample_decoder.Tensor( 'humans', shape_keys='humans/format'),
# mask_decoder = slim.tfexample_decoder.Image( 'mask/encoded', 'mask/format')
# maskd = mask_decoder.tensors_to_item(features)
# formats = features['mask/format']
# decoder =  slim.tfexample_decoder.Tensor('xx')
#xx = decoder2.tensors_to_item(features)

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(5):
        a,b = sess.run([tf_humans, tf_keypoints])
        print(a.shape)
        print(a)
        print(b)
        #plt.imshow(a)
        #plt.show()
    coord.request_stop()
coord.join(threads) 