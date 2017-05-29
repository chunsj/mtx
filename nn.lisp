(in-package :mtx)

(declaim (optimize (speed 3) (safety 0) (debug 1)))

(defclass SNN (NEURALNETWORK)
  ((layers :initform nil)
   (rlayers :initform nil)
   (errlayer :initform nil)
   (optimizer :initform nil)))

(defun $snn (lyrs &key (el ($mse-layer)) (o ($sgd-optimizer :lr 0.01)))
  (let ((nw (make-instance 'SNN)))
    (with-slots (layers rlayers errlayer optimizer) nw
      (setf layers lyrs)
      (setf rlayers (reverse layers))
      (setf errlayer el)
      (setf optimizer o)
      (let ((params nil))
        (dolist (l layers) (setf params (append params (parameters l))))
        (initialize-parameters optimizer params)))
    nw))

(defmethod predict ((nw SNN) &key xs train)
  (with-slots (layers) nw
    (let ((ys xs))
      (dolist (l layers)
        (setf ys (forward-propagate l :xs ys :train train)))
      ys)))

(defmethod loss ((nw SNN) &key xs ts)
  (with-slots (errlayer layers wdl) nw
    (let ((ys (predict nw :xs xs :train T))
          (wd 0.0))
      (dolist (l layers) (incf wd (regularization l)))
      ($+ (forward-propagate errlayer :ys ys :ts ts)
          wd))))

(defmethod gradient ((nw SNN) &key xs ts)
  (with-slots (errlayer rlayers) nw
    (loss nw :xs xs :ts ts)
    (let ((dout (backward-propagate errlayer)))
      (dolist (l rlayers)
        (setf dout (backward-propagate l :d dout))))))

(defmethod train ((nw SNN) &key xs ts)
  (with-slots (layers optimizer) nw
    (gradient nw :xs xs :ts ts)
    (dolist (l layers)
      (optimize-parameters l :o optimizer))))
