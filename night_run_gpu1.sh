#!/bin/bash

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python train.py roi_ps64_mm128_gauss_adam
THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python train.py roi_ps64_mm128_gauss_adam_big

