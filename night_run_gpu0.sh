#!/bin/bash
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py ije_roi_ps64_mm128_nodrop
THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python train.py ije_roi_ps64_mm128



