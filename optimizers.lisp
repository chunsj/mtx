(in-package :mtx)

(declaim (optimize (speed 3) (safety 0) (debug 1)))

(defclass SGD (OPTIMIZER)
  ((lr :initarg :lr :initform 0.01)))

(defun $sgd-optimizer (&key (lr 0.01)) (make-instance 'SGD :lr lr))

(defmethod update ((sgd SGD) params grads)
  (with-slots (lr) sgd
    (let ((alpha (* -1 lr)))
      (loop :for p :in params :for g :in grads
         :do ($axpy g p :alpha alpha)))))

(defclass MOMENTUM (OPTIMIZER)
  ((lr :initarg :lr :initform 0.01)
   (m :initarg :m :initform 0.9)
   (vs :initform nil)))

(defun $momentum-optimizer (&key (lr 0.01) (m 0.9))
  (make-instance 'MOMENTUM :lr lr :m m))

(defmethod initialize-parameters ((mo MOMENTUM) params)
  (with-slots (vs) mo
    (dolist (p params)
      (setf (getf vs p) ($m 0 ($nrow p) ($ncol p))))))

(defmethod update ((mo MOMENTUM) params grads)
  (with-slots (lr vs m) mo
    (let ((alpha (* -1 lr)))
      (loop :for p :in params :for g :in grads
         :do (let ((v (getf vs p)))
               ($axpy g ($scal m v) :alpha alpha)
               ($axpy v p))))))

(defclass NESTEROV (OPTIMIZER)
  ((lr :initarg :lr :initform 0.01)
   (m :initarg :m :initform 0.9)
   (vs :initform nil)))

(defun $nesterov-optimizer (&key (lr 0.01) (m 0.9))
  (make-instance 'NESTEROV :lr lr :m m))

(defmethod initialize-parameters ((n NESTEROV) params)
  (with-slots (vs) n
    (dolist (p params)
      (setf (getf vs p) ($m 0 ($nrow p) ($ncol p))))))

(defmethod update ((n NESTEROV) params grads)
  (with-slots (lr m vs) n
    (let ((alpha (* -1.0 lr))
          (m2 (* m m))
          (m1 (* (- 1.0 m) (* -1.0 lr))))
      (loop :for p :in params :for g :in grads
         :do (let ((v (getf vs p)))
               ($axpy g ($scal m v) :alpha alpha)
               ($axpy ($+ ($x m2 v) ($x m1 g)) p))))))

(defclass ADAGRAD (OPTIMIZER)
  ((lr :initarg :lr :initform 0.01)
   (hs :initform nil)))

(defun $adagrad-optimizer (&key (lr 0.01))
  (make-instance 'ADAGRAD :lr lr))

(defmethod initialize-parameters ((a ADAGRAD) params)
  (with-slots (hs) a
    (dolist (p params)
      (setf (getf hs p) ($m 0 ($nrow p) ($ncol p))))))

(defmethod update ((a ADAGRAD) params grads)
  (with-slots (lr hs) a
    (let ((alpha (* -1 lr)))
      (loop :for p :in params :for g :in grads
         :do (let ((h (getf hs p)))
               ($axpy ($x g g) h)
               ($axpy ($x ($map (lambda (v) (/ alpha (+ 1.0E-7 (sqrt v)))) h) g) p))))))

(defclass RMSPROP (OPTIMIZER)
  ((lr :initarg :lr :initform 0.01)
   (dr :initarg :dr :initform 0.99)
   (hs :initform nil)))

(defun $rmsprop-optimizer (&key (lr 0.01) (dr 0.99))
  (make-instance 'RMSPROP :lr lr :dr dr))

(defmethod initialize-parameters ((p RMSPROP) params)
  (with-slots (hs) p
    (dolist (p params)
      (setf (getf hs p) ($m 0 ($nrow p) ($ncol p))))))

(defmethod update ((p RMSPROP) params grads)
  (with-slots (lr dr hs) p
    (let ((alpha (* -1.0 lr))
          (dr1 (- 1.0 dr)))
      (loop :for p :in params :for g :in grads
         :do (let ((h (getf hs p)))
               ($axpy ($x g g) ($scal dr h) :alpha dr1)
               ($axpy ($x ($map (lambda (v) (/ alpha (+ 1.0E-7 (sqrt v)))) h) g) p))))))

(defclass ADAM (OPTIMIZER)
  ((lr :initarg :lr :initform 0.001)
   (b1 :initarg :b1 :initform 0.9)
   (b2 :initarg :b2 :initform 0.999)
   (it :initform 0)
   (ms :initform nil)
   (vs :initform nil)))

(defun $adam-optimizer (&key (lr 0.001) (b1 0.9) (b2 0.999))
  (make-instance 'ADAM :lr lr :b1 b1 :b2 b2))

(defmethod initialize-parameters ((a ADAM) params)
  (with-slots (ms vs) a
    (dolist (p params)
      (setf (getf ms p) ($m 0 ($nrow p) ($ncol p)))
      (setf (getf vs p) ($m 0 ($nrow p) ($ncol p))))))

(defmethod update ((a ADAM) params grads)
  (with-slots (lr b1 b2 it ms vs) a
    (let ((alpha (* -1 lr))
          (alphat nil))
      (setf it (1+ it))
      (setf alphat (/ (* alpha (sqrt (- 1.0 (expt b2 it)))) (- 1.0 (expt b1 it))))
      (loop :for p :in params :for g :in grads
         :do (let ((m (getf ms p))
                   (v (getf vs p)))
               ($axpy ($x (- 1.0 b1) ($- g m)) m)
               ($axpy ($x (- 1.0 b2) ($- ($expt g 2) v)) v)
               ($axpy ($map (lambda (mv vv) (/ (* alphat mv) (+ 1.0E-7 vv))) m v) p))))))
