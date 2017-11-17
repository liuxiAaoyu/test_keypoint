# TensorFlow-Slim models

## Add new op

tensorflow 1.2:

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -std=c++11 -shared draw_keypoints.cc -o draw_keypoints.so -fPIC -I $TF_INC -O2