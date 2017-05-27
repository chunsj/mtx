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

(defclass SIMPLEMNISTNET (NEURALNETWORK)
  ((params :initform nil)
   (affinelayer01 :initform nil)
   (sigmoidlayer01 :initform nil)
   (affinelayer02 :initform nil)
   (sigmoidlayer02 :initform nil)
   (lastlayer :initform nil)
   (layers :initform nil)
   (optimizer :initform nil :accessor optimizer)))

(defun $simple-mnist (input-size hidden-size output-size)
  (let ((nw (make-instance 'SIMPLEMNISTNET)))
    (with-slots (params
                 affinelayer01 sigmoidlayer01
                 affinelayer02 sigmoidlayer02
                 lastlayer layers)
        nw
      (setf params (list :W1 ($r input-size hidden-size)
                         :b1 ($r 1 hidden-size)
                         :W2 ($r hidden-size output-size)
                         :b2 ($r 1 output-size)))
      (setf affinelayer01 ($affine-layer params :W1 :b1))
      (setf sigmoidlayer01 ($relu-layer params))
      (setf affinelayer02 ($affine-layer params :W2 :b2))
      (setf sigmoidlayer02 ($relu-layer params))
      (setf lastlayer ($softmax-layer params))
      (setf layers (list affinelayer01
                         sigmoidlayer01
                         affinelayer02
                         sigmoidlayer02)))
    nw))

(defmethod predict ((nw SIMPLEMNISTNET) &key xs)
  (with-slots (layers) nw
    (let ((iv xs))
      (dolist (layer layers)
        (setf iv (forward-propagate layer :xs iv)))
      iv)))

(defmethod loss ((nw SIMPLEMNISTNET) &key xs ts)
  (with-slots (lastlayer) nw
    (forward-propagate lastlayer :ys (predict nw :xs xs) :ts ts)))

(defmethod gradient ((nw SIMPLEMNISTNET) &key xs ts)
  (with-slots (layers lastlayer affinelayer01 affinelayer02) nw
    (let ((rlayers (reverse layers)))
      (loss nw :xs xs :ts ts)
      (let ((dout (backward-propagate lastlayer)))
        (dolist (layer rlayers)
          (setf dout (backward-propagate layer :d dout))))
      (list :W1 (dw affinelayer01)
            :b1 (db affinelayer01)
            :W2 (dw affinelayer02)
            :b2 (db affinelayer02)))))

(defmethod train ((nw SIMPLEMNISTNET) &key xs ts)
  (let ((grads (gradient nw :xs xs :ts ts))
        (optimizer (optimizer nw)))
    (with-slots (params) nw
      (update optimizer params grads))
    nw))

;; XXX maybe there's a bug in softmaxceelayer
;; simple xor test, i cannot make this work (with mse-loss it works)
(let ((X ($m '((0 0) (1 0) (0 1) (1 1))))
      (y ($m '(0 1 1 0) 1))
      (nw ($simple-mnist 2 4 1))
      (ntr 1000))
  (dotimes (i ntr) (train nw :xs X :ts y :lr 1))
  (print ($round (predict nw :xs X))))

;; mnist application
(defun simple-mnist-app (o)
  (let ((X (getf *mnist* :train-images))
        (y (getf *mnist* :train-labels))
        (nw ($simple-mnist 784 50 10))
        (ntr 1))
    (when o (setf (optimizer nw) o))
    (print ($str "LOSS[0]: " (loss nw :xs X :ts y)))
    (finish-output)
    (dotimes (i ntr)
      (print ($str "TR: " (1+ i)))
      (finish-output)
      (train nw :xs X :ts y))
    (print ($str "LOSS[F]: " (loss nw :xs X :ts y)))
    (finish-output)))

(simple-mnist-app (make-instance 'SGD :lr 1.0))
(simple-mnist-app (make-instance 'MOMENTUM :lr 1.0 :m 0.9))
(simple-mnist-app (make-instance 'NESTEROV :lr 1.0 :m 0.9))
(simple-mnist-app (make-instance 'ADAGRAD :lr 1.0))
(simple-mnist-app (make-instance 'RMSPROP :lr 1.0 :dr 0.99))
(simple-mnist-app (make-instance 'ADAM :lr 1.0 :b1 0.9 :b2 0.999))
