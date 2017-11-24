import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from preprocessing import cmu_paf_preprocessing
#preprocess = preprocessing_factory.get_preprocessing('my_pre',is_training=True)

image_shape = [368,368]
image_file = tf.placeholder(tf.string)
#img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
img_input = tf.image.decode_jpeg(image_file,3)
in_keypoints = tf.placeholder(tf.float32)
in_humans = tf.placeholder(tf.float32)
#image_pre = preprocess(img_input, image_shape[0], image_shape[1])
image_pre = cmu_paf_preprocessing.preprocess_for_train2(img_input, image_shape[0], image_shape[1], in_humans, in_keypoints,None)
#image_4d = tf.expand_dims(image_pre, 0)
isess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

isess.run(tf.global_variables_initializer())

f = open('/home/xiaoyu/Documents/data/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json','r')
s = json.load(f)
for item in s[2:]:
    imgpath = '/home/xiaoyu/Documents/data/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902/'+item['image_id']+'.jpg'
    humans = []
    tl=list(item['human_annotations'].items())
    tl.sort()
    for key, value in tl:
        humans.append([value[1],value[0],value[3],value[2]])
    keypoints  = []
    tl=list(item['keypoint_annotations'].items())
    tl.sort()
    for key, value in tl:
        temppoints = []
        for i in range(14):
            temppoints.append([value[i*3], value[i*3+1], value[i*3+2]])
        keypoints.append(temppoints)
    k = np.asarray(keypoints)
    h = np.asarray(humans)

    tk = tf.constant(k,dtype=tf.float32)
    th = tf.constant(h,dtype=tf.float32)

    imgstring = tf.gfile.FastGFile(imgpath,'rb').read()
    image = cv2.imread(imgpath)

    hh = tf.shape(img_input)[0]
    ww = tf.shape(img_input)[1]
    sh = tf.stack([hh-1, ww-1, hh-1, ww-1])
    sh = tf.cast(sh,dtype=tf.float32)
    bbox = th/sh
    th = bbox
    bbox = tf.expand_dims(bbox,0)

    tk_last = tf.strided_slice(tk,[0,0,2],[3,14,3],[1,1,3])
    last_ones = tf.ones([3])
    if 1:
        tk_last_full = tk_last*last_ones
    else:
        tk_last_full = tk_last*last_ones*2
    tk_ones = tf.ones(tf.shape(tk))
    mask = tf.greater(tk_last_full,tk_ones)
    tk_zeros = tf.zeros(tf.shape(tk))
    tk_1 = tf.where(mask,tk_zeros,tk)
    sh = tf.stack([ww-1, hh-1, 1])
    sh = tf.cast(sh,dtype=tf.float32)
    tk = tk_1/sh
    imagei = tf.image.convert_image_dtype(img_input, dtype=tf.float32)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(imagei),
            bounding_boxes=bbox,
            min_object_covered=0.8,
            aspect_ratio_range=(0.8,1.2),
            area_range=(0.8,1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(imagei, 0), distort_bbox)
    cropped_image = tf.slice(imagei, bbox_begin, bbox_size)

    distort_bbox = distort_bbox[0,0]

    pad=tf.constant([[0,1]])
    tf.pad(distort_bbox,pad)
    #print(isess.run(tk,feed_dict={image_file:imgstring}))
    # draw_keypoints_module = tf.load_op_library('./draw_keypoints.so')
    # img_float = img_input/255
    # drawed = draw_keypoints_module.draw_keypoints(tf.expand_dims(img_float,0),tf.expand_dims(tk,0))
    # #print(drawed)
    # plt.imshow(isess.run(drawed,feed_dict={image_file:imgstring})[0])
    # plt.show()

    keypointsx, keypointsy, keypointsv = tf.split(tk,[1,1,1],axis=2)
    #check x, y 
    keypointsx_clamp_mask1 = tf.greater(keypointsx,distort_bbox[1])
    keypointsx_clamp_mask2 = tf.less(keypointsx,distort_bbox[3])
    keypointsx_clamp_mask = tf.logical_and(keypointsx_clamp_mask1, keypointsx_clamp_mask2)
    keypointsy_clamp_mask1 = tf.greater(keypointsy,distort_bbox[0])
    keypointsy_clamp_mask2 = tf.less(keypointsy,distort_bbox[2])
    keypointsy_clamp_mask = tf.logical_and(keypointsy_clamp_mask1, keypointsy_clamp_mask2)
    keypointsv_clamp_mask = tf.logical_and(keypointsx_clamp_mask, keypointsy_clamp_mask)
    keypoints_clamp_mask = tf.where(keypointsv_clamp_mask, tf.ones(tf.shape(keypointsx)), tf.zeros(tf.shape(keypointsx)))
    keypoints_clamp_mask = keypoints_clamp_mask * tf.ones([3])
    tk_clamp = tk * keypoints_clamp_mask
    #print(isess.run(tk_clamp*sh,feed_dict={image_file:imgstring}))

    draw_keypoints_module = tf.load_op_library('./draw_keypoints.so')
    drawed = draw_keypoints_module.draw_keypoints( image_with_distorted_box, tf.expand_dims(tk_clamp, 0))


    box_ref = tf.stack([ distort_bbox[3] - distort_bbox[1], distort_bbox[2] - distort_bbox[0], 1])
    box_v = tf.stack([ distort_bbox[1], distort_bbox[0], 0])
    tk_clamp = tk_clamp - box_v
    tk_clamp = tk_clamp / box_ref
    #print(isess.run(tk_clamp,feed_dict={image_file:imgstring}))
    drawed_croped = draw_keypoints_module.draw_keypoints( tf.expand_dims(cropped_image, 0), tf.expand_dims(tk_clamp, 0))

    humans_ymin, humans_xmin, humans_ymax, humans_xmax = tf.split(th, [1, 1, 1, 1], axis=1)
    area_human = (humans_ymax - humans_ymin)*(humans_xmax - humans_xmin)#tf.multiply(tf.subtract(humans_ymax, humans_ymin), tf.subtract(humans_xmax, humans_xmin))
    area_factor = area_human / ((distort_bbox[3] - distort_bbox[1]) * (distort_bbox[2] - distort_bbox[0]))

    #print(isess.run([area_factor, ((distort_bbox[3] - distort_bbox[1]) * (distort_bbox[2] - distort_bbox[0])), th], feed_dict={image_file:imgstring}))


    put_gaussian_maps_module = tf.load_op_library('./put_gaussian_maps.so')
    gaussion_maps = put_gaussian_maps_module.put_gaussian_maps( cropped_image, tk_clamp)
    #gaussion_maps = tf.squeeze(gaussion_maps,[0])

    put_vec_maps_module = tf.load_op_library('./put_vec_maps.so')
    vec_maps = put_vec_maps_module.put_vec_maps( cropped_image, tk_clamp, area_factor)
    #vec_maps = tf.squeeze(vec_maps,[0])
    show_gaussion_maps, show_vec_maps, show_cropped = isess.run([gaussion_maps, vec_maps, drawed_croped], feed_dict={image_file:imgstring})

    ttt = np.zeros(show_vec_maps[:,:,0].shape)
    for i in range(14):
        ttt = ttt + show_vec_maps[:,:,i*2]
    plt.imshow(show_cropped[0])
    plt.imshow(ttt,alpha = .6)
    plt.show()
    plt.imshow(ttt)
    plt.show()

    ttt = np.zeros(show_gaussion_maps[:,:,0].shape)
    for i in range(14):
        ttt = ttt + show_gaussion_maps[:,:,i]
    plt.imshow(show_cropped[0])
    plt.imshow(ttt,alpha = .6)
    plt.show()

    plt.imshow(ttt)
    plt.show()








# def cond(t, output, i, j):
#     return tf.less(i, tf.shape(t)[0])

# def body(t, output, i, j):
#     output = output.write(i, tf.add(t[i], 10))
#     return t, output, i + 1

# # TensorArray is a data structure that support dynamic writing
# output_ta = tf.TensorArray(dtype=tf.float32,
#                size=0,
#                dynamic_size=True,
#                element_shape=(x.get_shape()[1],))
# _, output_op, _  = tf.while_loop(cond, body, [tk, output_ta, 0, 0])
# output_op = output_op.stack()

