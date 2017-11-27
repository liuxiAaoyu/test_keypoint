from preprocessing import preprocessing_factory
from preprocessing import cmu_paf_preprocessing
from nets import nets_factory
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import math
import cv2

#preprocess = preprocessing_factory.get_preprocessing('my_pre',is_training=True)
DATA_PATH = '/media/xiaoyu/Document/data/'
DATA_PATH = '/home/Documents/data/'
#/home/xiaoyu/Documents/data/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json
#/home/xiaoyu/Documents/data/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911 (copy).json

image_shape = [299,299]
image_file = tf.placeholder(tf.string)
#img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
img_input = tf.image.decode_jpeg(image_file,3)
in_keypoints = tf.placeholder(tf.float32)
in_humans = tf.placeholder(tf.float32)
#image_pre = preprocess(img_input, image_shape[0], image_shape[1])
image_pre = cmu_paf_preprocessing.preprocess_for_eval(img_input, image_shape[0], image_shape[1])
image_4d = tf.expand_dims(image_pre, 0)

network_fn = nets_factory.get_network('cmu_paf_net', is_training=False)
gaussian_out, vec_out= network_fn(image_4d)
gaussian_out = tf.image.resize_bilinear(gaussian_out[5], [image_shape[0], image_shape[1]],align_corners=False)
gaussian_out = tf.squeeze(gaussian_out, [0])
vec_out = tf.image.resize_bilinear(vec_out[5], [image_shape[0], image_shape[1]],align_corners=False)
vec_out = tf.squeeze(vec_out, [0])

isess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

isess.run(tf.global_variables_initializer())
ckpt_filename = '/home/xiaoyu/Documents/test_keypoint/log/model.ckpt-24016'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)


f = open('/media/xiaoyu/Document/data/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json','r')
s = json.load(f)

for item in s:
    imgpath = '/media/xiaoyu/Document/data/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902/'+item['image_id']+'.jpg'
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
            temppoints.append([value[i*3],value[i*3+1],value[i*3+2]])
        keypoints.append(temppoints)
    k = np.asarray(keypoints)
    h = np.asarray(humans)
    imgstring = tf.gfile.FastGFile(imgpath,'rb').read()


    show_image, show_gaussian, show_vec = isess.run([ image_pre, gaussian_out, vec_out],
                                                            feed_dict={image_file:imgstring})
    show_image = ((show_image/2+0.5))
 

    ttt = np.zeros(show_vec[:,:,0].shape)
    for i in range(14):
        ttt = ttt + show_vec[:,:,i*2]
    plt.imshow(show_image)
    plt.imshow(ttt,alpha = .6)
    plt.show()

    ttt = np.zeros(show_gaussian[:,:,0].shape)
    for i in range(14):
        ttt = ttt + show_gaussian[:,:,i]
    plt.imshow(show_image)
    plt.imshow(ttt,alpha = .6)
    plt.show()

    plt.imshow(ttt)
    plt.show()





    oriImg = show_image
    all_peaks = []
    peak_counter = 0

    for part in range(14):
        map_ori = show_gaussian[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)
        
        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
        
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > 0.1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)


    #mid_1[14] = {12, 13, 0, 1, 13, 3, 4, 0, 6, 7, 3, 9, 10, 6};
    #mid_2[14] = {13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 9};
    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[12,13], [13,0], [0,1], [1,2], [13,3], [3,4], [4,5], [0,6], [6,7], \
            [7,8], [3,9], [9,10], [10,11], [6,9]]
    # the middle joints heatmap correpondence
    mapIdx = [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15], \
            [16,17], [18,19], [20,21], [22,23], [24,25], [26,27]]

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = show_vec[:,:,[x for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]]
        candB = all_peaks[limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    vec = np.divide(vec, norm)
                    
                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                    
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                    for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                    for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > 0.05 )[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k])

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                
                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print ("found = 2")
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])


    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)                    



    # visualize
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')

    #canvas = cv2.imread(test_image) # B,G,R order
    canvas = oriImg
    for i in range(14):
        rgba = np.array(cmap(1 - i/18. - 1./36))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    to_plot = cv2.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:,:,[2,1,0]])
    plt.show()


    # visualize 2
    stickwidth = 4

    for i in range(14):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            
    plt.imshow(canvas[:,:,[2,1,0]])
    plt.show()
