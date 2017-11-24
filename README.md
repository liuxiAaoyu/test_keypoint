# TensorFlow-Slim models

## Add new op

tensorflow 1.2:

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -std=c++11 -shared draw_keypoints.cc -o draw_keypoints.so -fPIC -I $TF_INC -O2

tensorflow 1.4:
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared put_gaussian_maps.cc -o put_gaussian_maps.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
g++ -std=c++11 -shared put_vec_maps.cc -o put_vec_maps.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
