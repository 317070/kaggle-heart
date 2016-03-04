#!/bin/bash

#THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python train.py gauss_roi_zoom_big
#THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi_zoom_big

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python predict_framework_transfer.py gauss_roi_zoom_mask_leaky_after 50 arithmetic
THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python predict_framework_transfer.py meta_gauss_roi_zoom_mask_leaky_after 50 arithmetic

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python predict_framework_transfer.py gauss_roi_zoom_big 50 arithmetic
THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python predict_framework_transfer.py meta_gauss_roi_zoom_big 50 arithmetic

#THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python predict_framework_transfer.py gauss_roi_zoom 50 arithmetic
