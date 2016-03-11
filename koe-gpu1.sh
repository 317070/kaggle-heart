cd ira
THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once' python train.py gauss_roi_zoom_mask_leaky

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once' python train_meta.py meta_gauss_roi_zoom_mask_leaky

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once' python predict_framework_transfer.py gauss_roi_zoom_mask_leaky 50 arithmetic

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once' python predict_framework_transfer.py meta_gauss_roi_zoom_mask_leaky 50 arithmetic

---

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once' python train.py gauss_roi_zoom

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once' python train_meta.py meta_gauss_roi_zoom

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once' python predict_framework_transfer.py gauss_roi_zoom 50 arithmetic

THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once' python predict_framework_transfer.py meta_gauss_roi_zoom 50 arithmetic
