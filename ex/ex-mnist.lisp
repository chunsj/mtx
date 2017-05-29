(in-package :mtx)

(defparameter *mnist* (read-mnist-data))
(defparameter *mnist-train-indices* ($shuffle ($iota ($nrow (getf *mnist* :train-images)))))
(defparameter *mnist-test-indices* ($shuffle ($iota ($nrow (getf *mnist* :test-images)))))

;; batch data generation
(let* ((xtrain (getf *mnist* :train-images))
       (bsize 100)
       (index-parts ($partition *mnist-train-indices* bsize))
       (bindices (car index-parts))
       (xbatch ($rows xtrain bindices)))
  (print bindices)
  (print ($dim xbatch)))

(let* ((train-images (getf *mnist* :train-images))
       (train-labels (getf *mnist* :train-labels))
       (test-images (getf *mnist* :test-images))
       (test-labes (getf *mnist* :test-labels))
       (train-samples (min 300 ($nrow (getf))))
       (test-samples 100)
       (X ($rows (getf *mnist* :train-images) ($iota train-samples)))
       (y ($rows (getf *mnist* :train-labels) ($iota train-samples)))
       (tX ($rows (getf *mnist* :test-images) ($iota test-samples)))
       (ty ($rows (getf *mnist* :test-labels) ($iota test-samples)))
       (input-size ($ncol ($ X 0 T)))
       (hidden-size 50)
       (output-size 10)
       (layers (list ($affine-layer input-size hidden-size :winit :xavier)
                     ($tanh-layer)
                     ($affine-layer hidden-size output-size :winit :xavier)))
       (o ($sgd-optimizer :lr 0.1))
       (nn ($snn layers :el ($softmax-layer) :o o))
       (ntr 2000))
  (print ($str "INITIAL: " ($argmax (predict nn :xs X) :axis :row)))
  (time
   (dotimes (i ntr)
     (train nn :xs X :ts y)
     (when (= 0 (rem i 100))
       (print ($str "ITR[" i "] TRAIN ERROR: " (loss nn :xs X :ts y)))
       (print ($str "            TEST ERROR: " (loss nn :xs tX :ts ty)))
       (finish-output))))
  (print ($str "FINAL: " ($argmax (predict nn :xs X) :axis :row)))
  (print ($str "TRUE: " ($argmax y :axis :row)))
  (print ($str "FINAL[T]: " ($argmax (predict nn :xs tX) :axis :row)))
  (print ($str "TRUE[T]: " ($argmax ty :axis :row))))
