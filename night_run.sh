#!/bin/bash
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py conv_rms_sys
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py conv_rms_dst
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py conv_rms_sys_nonorm
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py conv_rms_dst_nonorm
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py conv_rms_sys_dst
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py vgg_rms_sd
