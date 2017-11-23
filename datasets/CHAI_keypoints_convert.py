import tensorflow as tf
import numpy as np
import glob
import os
import random
from dataset_utils import int64_feature, float_feature, bytes_feature
import matplotlib.pyplot as plt
import json
import os

DATA_PATH = '/media/xiaoyu/Document/data/ai_challenger_keypoint_train_20170909/'
#DATA_PATH = '/home/xiaoyu/Documents/data/ai_challenger_keypoint_validation_20170911/'
#/home/xiaoyu/Documents/data/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json
#/home/xiaoyu/Documents/data/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json
def process_data(image, humans, keypoints, tfrecord_write):
    image_format = b'JPEG'
    example = tf.train.Example( features = tf.train.Features(
        feature ={
            'image/encoded' : bytes_feature(image),
            'image/format' : bytes_feature(image_format),
            'humans' :float_feature( humans.flatten().tolist()),
            'humans/shape' :int64_feature(np.array(humans.shape).flatten().tolist()),
            'keypoints' :float_feature(keypoints.flatten().tolist()),
            'keypoints/shape' :int64_feature(np.array(keypoints.shape).flatten().tolist()),
        }
    ))
    tfrecord_write.write(example.SerializeToString())

f = open(DATA_PATH+'keypoint_train_annotations_20170909.json','r')
s = json.load(f)
count=0
with tf.Session() as sess:
    with tf.python_io.TFRecordWriter('/home/xiaoyu/Documents/data/train.tfrecord') as tfrecord_writer:
        for item in s:
            imgpath = DATA_PATH+'keypoint_train_images_20170902/'+item['image_id']+'.jpg'
            if os.path.exists(imgpath) == False:
                continue
            humans = []
            tl=list(item['human_annotations'].items())
            tl.sort()
            flag = 0
            for key, value in tl:
                if value[2] == value[0] or value[3] == value[1]:
                    flag = 1
                humans.append([value[1],value[0],value[3],value[2]])
            if flag == 1:
                continue
            keypoints  = []
            tl=list(item['keypoint_annotations'].items())
            tl.sort()
            for key, value in tl:
                temppoints = []
                for i in range(14):
                    temppoints.append([value[i*3],value[i*3+1],value[i*3+2]])
                keypoints.append(temppoints)

            k = np.asarray(keypoints)
            h = np.asarray(humans)
            image = tf.gfile.FastGFile(imgpath,'rb').read()
            process_data(image, h, k, tfrecord_writer)
            count+=1
            print(count)
        print('train finished')