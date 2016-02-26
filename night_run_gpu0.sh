#!/bin/bash
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py gauss_roi
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train_meta.py  meta_jeroen_gauss_roi_w_notrain gauss_roi-geit-20160225-224741.pkl



