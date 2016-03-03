#!/bin/bash
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py gauss_roi_zoom
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi_zoom

THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py gauss_roi_zoom_big
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi_zoom_big

THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py gauss_roi_zoom_mask_leaky_after
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train_meta.py meta_gauss_roi_zoom_mask_leaky_after


