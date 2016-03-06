#!/bin/bash

#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py gauss_roi10_big_leaky_after_seqshift
#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi10_big_leaky_after_seqshift
#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python predict_framework_transfer.py meta_gauss_roi10_big_leaky_after_seqshift 50 arithmetic


#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py gauss_roi10_maxout
#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi10_maxout
#THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python predict_framework_transfer.py meta_gauss_roi10_maxout 50 arithmetic


THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py ch2_zoom_leaky_nomask_seqshift
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python predict_framework_transfer.py ch2_zoom_leaky_nomask_seqshift 50 geometric

THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py ch4_zoom_leaky_nomask_seqshift
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python predict_framework_transfer.py ch4_zoom_leaky_nomask_seqshift 50 geometric

