(cl:defpackage :mtx
  (:use #:common-lisp
        #:fnv
        #:cl-blapack)
  (:export #:$size
           #:$dim
           #:$nrow
           #:$ncol
           #:$m
           #:$r
           #:$rn
           #:$ones
           #:$zeros
           #:$eye
           #:$
           #:$rows
           #:$cols
           #:$dup
           #:$reshape
           #:$transpose
           #:$sm
           #:$rot180
           #:$+
           #:$-
           #:$*
           #:$x
           #:$/
           #:$sum
           #:$max
           #:$min
           #:$argmax
           #:$argmin
           #:$mean
           #:$convolute
           #:$xwpb
           #:$gemm
           #:$mm
           #:$axpy
           #:$map
           #:$map!
           #:$round
           #:$exp
           #:$log
           #:$log10
           #:$expt
           #:$sqrt
           #:$sigmoid
           #:$sin
           #:$cos
           #:$tan
           #:$sinh
           #:$cosh
           #:$tanh
           #:$asin
           #:$acos
           #:$atan
           #:$softmax
           #:$relu
           #:$lkyrelu
           #:$mse
           #:$cee
           #:$shuffle
           #:$iota
           #:$linspace
           #:$partition
           #:@>
           #:@>>
           #:LAYER
           #:OPTIMIZER
           #:NEURALNETWORK
           #:forward-propagate
           #:backward-propagate
           #:optimize-parameters
           #:regularization
           #:parameters
           #:initialize-parameters
           #:update
           #:predict
           #:loss
           #:gradient
           #:train
           #:SIGMOIDLAYER
           #:$sigmoid-layer
           #:RELULAYER
           #:$relu-layer
           #:LKYRELULAYER
           #:$lkyrelu-layer
           #:SOFTMAXCEELOSSLAYER
           #:$softmax-layer
           #:PLAINMSELOSSLAYER
           #:$mse-layer
           #:DROPOUTLAYER
           #:$dropout-layer
           #:AFFINELAYER
           #:$affine-layer
           #:BATCHNORMLAYER
           #:batchnorm-layer
           #:SNN
           #:SGD
           #:$sgd-optimizer
           #:MOMENTUM
           #:$momentum-optimizer
           #:NESTEROV
           #:$nesterov-optimizer
           #:ADAGRAD
           #:$adagrad-optimizer
           #:RMSPROP
           #:$rmsprop-optimizer
           #:ADAM
           #:$adam-optimizer))
