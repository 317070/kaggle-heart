#!/bin/bash


#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train.py ch4_roi10_zoom_leaky_after_nomask
#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train.py ch2_roi10_zoom_leaky_after_nomask
#
#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train.py ch2_zoom_leaky_after_nomask
#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train.py ch4_zoom_leaky_after_nomask
#
#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train.py ch4_zoom_leaky_after
#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train.py ch2_zoom_leaky_after

#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python predict_framework_transfer.py ch4_roi10_zoom_leaky_after_nomask 50 arithmetic
#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python predict_framework_transfer.py ch2_roi10_zoom_leaky_after_nomask 50 arithmetic
#
#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python predict_framework_transfer.py ch2_zoom_leaky_after_nomask 50 arithmetic
#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python predict_framework_transfer.py ch4_zoom_leaky_after_nomask 50 arithmetic

#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python predict_framework_transfer.py gauss_roi10_zoom_mask_leaky_after 50 geometric
#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python predict_framework_transfer.py gauss_roi_zoom_mask_leaky 50 geometric
#THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python predict_framework_transfer.py gauss_roi10_big_leaky_after_seqshift 50 geometric


THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train.py gauss_roi10_maxout_seqshift_96
THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi10_maxout_seqshift_96
THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python predict_framework_transfer.py meta_gauss_roi10_maxout_seqshift_96 50 arithmetic

