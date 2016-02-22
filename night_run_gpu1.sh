#!/bin/bash

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python train_meta.py meta_roi_ps64_mm128_gauss_big_adam roi_ps64_mm128_gauss_big_adam-paard-20160221-150028.pkl