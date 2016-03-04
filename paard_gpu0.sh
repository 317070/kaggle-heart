#!/bin/bash

#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py gauss_roi_zoom_mask_leaky
#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi_zoom_mask_leaky

#paard
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python predict.py gauss_roi_zoom_mask_leaky valid 1
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python predict.py gauss_roi10_zoom_mask_leaky_after valid 1
