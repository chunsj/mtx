(cl:defpackage :mtx
  (:use #:common-lisp
        #:fnv
        #:cl-blapack)
  (:export #:size
           #:dim
           #:$m
           #:$r
           #:$rn
           #:$
           #:$rows
           #:$cols
           #:$dup
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
           #:$mse
           #:$cee
           #:$shuffle
           #:$iota
           #:$linspace
           #:$partition
           #:@>
           #:@>>
           #:LAYER
           #:forward-propagate
           #:backward-propagate
           #:SIGMOIDLAYER
           #:RELULAYER
           #:AFFINELAYER
           #:SOFTMAXCEELOSSLAYER
           #:PLAINMSELOSSLAYER
           #:NEURALNETWORK
           #:predict
           #:loss
           #:gradient
           #:train))
