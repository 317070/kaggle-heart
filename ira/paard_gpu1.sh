#!/bin/bash

THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train.py gauss_roi10_zoom_mask_leaky_after

