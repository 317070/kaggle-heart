#!/bin/bash
THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train.py roi_ps64_mm128_gauss
THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train.py roi_ps64_mm128_gauss_big




