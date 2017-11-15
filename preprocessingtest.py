from preprocessing import preprocessing_factory
from preprocessing import inception_preprocessing1
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#preprocess = preprocessing_factory.get_preprocessing('my_pre',is_training=True)

image_shape = [368,368]
image_file = tf.placeholder(tf.string)
#img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
img_input = tf.image.decode_jpeg(image_file,3)
in_keypoints = tf.placeholder(tf.float32)
in_humans = tf.placeholder(tf.float32)
#image_pre = preprocess(img_input, image_shape[0], image_shape[1])
bbox,image_pre = inception_preprocessing1.preprocess_for_train2(img_input, image_shape[0], image_shape[1], in_humans, in_keypoints, None)
image_4d = tf.expand_dims(image_pre, 0)
isess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

isess.run(tf.global_variables_initializer())

f = open('/home/xiaoyu/Documents/data/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json','r')
s = json.load(f)

for item in s:
    imgpath = '/home/xiaoyu/Documents/data/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/'+item['image_id']+'.jpg'
    humans = []
    for key, value in item['human_annotations'].items():
        humans.append([value[1],value[0],value[3],value[2]])
    keypoints  = []
    for key, value in item['keypoint_annotations'].items():
        temppoints = []
        for i in range(14):
            temppoints.append([value[i*3],value[i*3+1],value[i*3+2]])
        keypoints.append(temppoints)
    k = np.asarray(keypoints)
    h = np.asarray(humans)
    imgstring = tf.gfile.FastGFile(imgpath,'rb').read()
    boxs, image = isess.run([bbox,image_pre],feed_dict={image_file:imgstring, in_humans:h, in_keypoints:k})
    image = ((image/2+0.5))
    print(boxs)
    plt.imshow(image[0])
    plt.show()




