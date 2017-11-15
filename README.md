# TensorFlow-Slim models

## Add new op

tensorflow 1.2:
g++ -std=c++11 -shared draw_keypoints.cc -o draw_keypoints.so -fPIC -I $TF_INC -O2