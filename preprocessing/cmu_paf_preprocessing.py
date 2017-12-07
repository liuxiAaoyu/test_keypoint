# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.9,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.95, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox

draw_keypoints_module = tf.load_op_library('./draw_keypoints.so')
put_gaussian_maps_module = tf.load_op_library('./put_gaussian_maps.so')
put_vec_maps_module = tf.load_op_library('./put_vec_maps.so')
def preprocess_for_train2(image, height, width, hunmans, keypoints, bbox,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Additionally it would create image_summaries to display the different
  transformations applied to the image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations (i.e.
      bi-cubic resizing, random_hue or random_contrast).
    scope: Optional scope for name_scope.
    add_image_summaries: Enable image summaries.
  Returns:
    3-D float Tensor of distorted image used for training with range [-1, 1].
  """
  with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
    if bbox is None:
      bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])
    hh = tf.shape(image)[0]
    ww = tf.shape(image)[1]
    sh = tf.stack([hh, ww, hh, ww])
    sh = tf.cast(sh,dtype=tf.float32)
    th = hunmans/sh

    tk = keypoints
    tk_last = tf.strided_slice(tk,[0,0,2],[9999,14,3],[1,1,3])
    last_ones = tf.ones([3])
    if 1:
      tk_last_full = tk_last*last_ones
    else:
      tk_last_full = tk_last*last_ones*2
    tk_ones = tf.ones(tf.shape(tk))
    mask = tf.greater(tk_last_full,tk_ones)
    tk_zeros = tf.zeros(tf.shape(tk))
    tk_1 = tf.where(mask,tk_zeros,tk)
    sh = tf.stack([ww, hh, 1])
    sh = tf.cast(sh,dtype=tf.float32)
    tk = tk_1/sh

    #bbox = tf.expand_dims(th,0)
    #hunmans
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox)
    if add_image_summaries:
      tf.summary.image('image_with_bounding_boxes', image_with_box)

    distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
    
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), distorted_bbox)
    if add_image_summaries:
      tf.summary.image('images_with_distorted_bounding_box',
                       image_with_distorted_box)
    #return distorted_bbox,image_with_distorted_box

    keypointsx, keypointsy, keypointsv = tf.split(tk,[1,1,1],axis=2)
    distort_bbox = distorted_bbox[0,0]
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

    box_ref = tf.stack([ distort_bbox[3] - distort_bbox[1], distort_bbox[2] - distort_bbox[0], 1])
    box_v = tf.stack([ distort_bbox[1], distort_bbox[0], 0])
    tk_clamp = tk_clamp - box_v
    tk_clamp = tk_clamp / box_ref

    
    image_with_distorted_keypoints = draw_keypoints_module.draw_keypoints( tf.expand_dims(distorted_image, 0), tf.expand_dims(tk_clamp, 0))
    if add_image_summaries:
      tf.summary.image('cropped_image_with_keypoints',
                       image_with_distorted_keypoints)

    humans_ymin, humans_xmin, humans_ymax, humans_xmax = tf.split(th, [1, 1, 1, 1], axis=1)
    area_human = (humans_ymax - humans_ymin)*(humans_xmax - humans_xmin)#tf.multiply(tf.subtract(humans_ymax, humans_ymin), tf.subtract(humans_xmax, humans_xmin))
    area_factor = area_human / ((distort_bbox[3] - distort_bbox[1]) * (distort_bbox[2] - distort_bbox[0]))

    
    #gaussian_maps = put_gaussian_maps_module.put_gaussian_maps( tf.expand_dims(distorted_image, 0), tf.expand_dims(tk_clamp, 0))
    #gaussian_maps = tf.squeeze(gaussian_maps,0)
    gaussian_maps = put_gaussian_maps_module.put_gaussian_maps(distorted_image, tk_clamp)

    
    #vec_maps = put_vec_maps_module.put_vec_maps( tf.expand_dims(distorted_image, 0), tf.expand_dims(tk_clamp, 0), tf.expand_dims(area_factor, 0))
    #vec_maps = tf.squeeze(vec_maps,0)
    vec_maps = put_vec_maps_module.put_vec_maps( distorted_image, tk_clamp, area_factor)
    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.

    # We select only 1 case for fast_mode bilinear.
    num_resize_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=num_resize_cases)

    gaussian_maps = tf.image.resize_images(gaussian_maps, [35, 35], 0)
    vec_maps = tf.image.resize_images(vec_maps, [35, 35], 0)

    if add_image_summaries:
      tf.summary.image('cropped_resized_image',
                       tf.expand_dims( distorted_image, 0))
      tf.summary.image('cropped_resized_gaussian',
                       tf.expand_dims(tf.expand_dims(tf.reduce_sum(gaussian_maps,2),2), 0))
      tf.summary.image('cropped_resized_vec',
                       tf.expand_dims(tf.expand_dims(tf.reduce_sum(vec_maps,2),2), 0))



    # Randomly flip the image horizontally.
    #distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors. There are 4 ways to do it.
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering, fast_mode),
        num_cases=4)

    if add_image_summaries:
      tf.summary.image('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))
    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)

    return distorted_image, gaussian_maps, vec_maps


def preprocess_for_train(image, height, width, hunmans, keypoints, bbox,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Additionally it would create image_summaries to display the different
  transformations applied to the image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations (i.e.
      bi-cubic resizing, random_hue or random_contrast).
    scope: Optional scope for name_scope.
    add_image_summaries: Enable image summaries.
  Returns:
    3-D float Tensor of distorted image used for training with range [-1, 1].
  """
  with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
    if bbox is None:
      bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    hh = tf.shape(image)[0]
    ww = tf.shape(image)[1]
    sh = tf.stack([hh, ww, hh, ww])
    sh = tf.cast(sh,dtype=tf.float32)
    th = hunmans/sh
    humans_ymin, humans_xmin, humans_ymax, humans_xmax = tf.split(th, [1, 1, 1, 1], axis=1)
    area_human = (humans_ymax - humans_ymin)*(humans_xmax - humans_xmin)

    tk = keypoints
    tk_last = tf.strided_slice(tk,[0,0,2],[9999,14,3],[1,1,3])
    last_ones = tf.ones([3])
    if 1:
      tk_last_full = tk_last*last_ones
    else:
      tk_last_full = tk_last*last_ones*2
    tk_ones = tf.ones(tf.shape(tk))
    mask = tf.greater(tk_last_full,tk_ones)
    tk_zeros = tf.zeros(tf.shape(tk))
    tk_1 = tf.where(mask,tk_zeros,tk)
    sh = tf.stack([ww, hh, 1])
    sh = tf.cast(sh,dtype=tf.float32)
    tk = tk_1/sh
    
    gaussian_maps = put_gaussian_maps_module.put_gaussian_maps(image, tk)
    vec_maps = put_vec_maps_module.put_vec_maps( image, tk, area_human)

    image_with_mask = tf.concat([image, gaussian_maps, vec_maps], axis=2)
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox)
    if add_image_summaries:
      tf.summary.image('image_with_bounding_boxes', image_with_box)

    distorted_image_with_mask, distorted_bbox = distorted_bounding_box_crop(image_with_mask, bbox)

    distorted_image, distorted_gaussian_maps, distorted_vec_maps  = tf.split(distorted_image_with_mask, [3, 14, 30], axis=2)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), distorted_bbox)
    if add_image_summaries:
      tf.summary.image('images_with_distorted_bounding_box',
                       image_with_distorted_box)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.

    # We select only 1 case for fast_mode bilinear.
    num_resize_cases = 1 if fast_mode else 4
    distorted_image_with_mask = apply_with_random_selector(
        distorted_image_with_mask,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=num_resize_cases)
    
    distorted_image, distorted_gaussian_maps, distorted_vec_maps  = tf.split(distorted_image_with_mask, [3, 14, 30], axis=2)
    if add_image_summaries:
      tf.summary.image('cropped_resized_image',
                       tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    #distorted_image = tf.image.random_flip_left_right(distorted_image)

    uniform_random = tf.random_uniform([], 0, 1.0, seed=None)
    mirror_cond = tf.less(uniform_random, .5)
    def flip_left_right(image):
      filpfactor = tf.constant([1,1,1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,  
                        -1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,1,1],dtype=tf.float32)
      image = tf.reverse(image*filpfactor, [1])
      d_image, d_gaussian, d_vec  = tf.split(image, [3, 14, 30], axis=2)
      l_arm, r_arm, l_leg, r_leg, hn  = tf.split(d_gaussian, [3, 3, 3, 3, 2], axis=2)
      d_gaussian = tf.concat([ r_arm, l_arm, r_leg, l_leg, hn], axis=2)
      image = tf.concat([d_image, d_gaussian, d_vec], axis=2)
      return image
    distorted_image_with_mask_flip = tf.cond(mirror_cond,
                              lambda: flip_left_right(distorted_image_with_mask),
                              lambda: distorted_image_with_mask)
    distorted_image_with_mask_flip.set_shape(distorted_image_with_mask.get_shape())
    distorted_image, distorted_gaussian_maps, distorted_vec_maps  = tf.split(distorted_image_with_mask_flip, [3, 14, 30], axis=2)

    distorted_gaussian_maps = tf.image.resize_images(distorted_gaussian_maps, [35, 35], 0)
    distorted_vec_maps = tf.image.resize_images(distorted_vec_maps, [35, 35], 0)
    
    if add_image_summaries:
      tf.summary.image('cropped_resized_gaussian',
                       tf.expand_dims(tf.expand_dims(tf.reduce_sum(distorted_gaussian_maps,2),2), 0))
      tf.summary.image('cropped_resized_vec',
                       tf.expand_dims(tf.expand_dims(tf.reduce_sum(distorted_vec_maps,2),2), 0))

    # Randomly distort the colors. There are 4 ways to do it.
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering, fast_mode),
        num_cases=4)

    if add_image_summaries:
      tf.summary.image('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))
    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)
    return distorted_image, distorted_gaussian_maps, distorted_vec_maps



def preprocess_for_eval(image, height, width,
                        central_fraction=1.0, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def preprocess_image(image, height, width,
                     hunmans, keypoints,
                     is_training=False,
                     bbox=None,
                     fast_mode=True,
                     add_image_summaries=True):
  """Pre-process one image for training or evaluation.

  Args:
    image: 3-D Tensor [height, width, channels] with the image. If dtype is
      tf.float32 then the range should be [0, 1], otherwise it would converted
      to tf.float32 assuming that the range is [0, MAX], where MAX is largest
      positive representable number for int(8/16/32) data type (see
      `tf.image.convert_image_dtype` for details).
    height: integer, image expected height.
    width: integer, image expected width.
    is_training: Boolean. If true it would transform an image for train,
      otherwise it would transform it for evaluation.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations.
    add_image_summaries: Enable image summaries.

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
  if is_training:
    return preprocess_for_train(image, height, width, hunmans, keypoints, bbox, fast_mode,
                                add_image_summaries=add_image_summaries)
  else:
    return preprocess_for_eval(image, height, width)
