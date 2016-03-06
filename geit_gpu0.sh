#!/bin/bash

#geit
#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi_zoom_mask_leaky_after
#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py gauss_roi_zoom
#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi_zoom

THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py gauss_roi_zoom_big_after_seqshift
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi_zoom_big_after_seqshift
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python predict_framework_transfer.py gauss_roi_zoom_big_after_seqshift 50 arithmetic
