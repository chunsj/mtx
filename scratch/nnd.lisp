(in-package :mtx)

(defclass AFL ()
  ((input-size :initarg :input-size :initform nil)
   (output-size :initarg :output-size :initform nil)
   (w :initform nil)
   (b :initform nil)
   (x :initform nil)
   (dw :reader dw :initform nil)
   (db :reader db :initform nil)))

(defmethod print-object ((l AFL) stream)
  (print-unreadable-object (l stream :type t :identity t)
    (with-slots (input-size output-size) l
      (format stream "[~A x ~A]" input-size output-size))))

(defun $afl (input-size output-size &key winit)
  (let ((l (make-instance 'AFL :input-size input-size :output-size output-size)))
    (with-slots (w b dw db) l
      (let ((sc (cond ((eq winit :he) (sqrt (/ 6.0 input-size)))
                      ((eq winit :xavier) (sqrt (/ 3.0 input-size)))
                      (T 1.0))))
        (setf w ($x sc ($- ($x 2.0 ($r input-size output-size)) 1.0))
              dw ($m 0.0 input-size output-size))
        (setf b ($m 1.0 1 output-size)
              db ($m 0.0 1 output-size)))
      l)))

(defmethod forward-propagate ((l AFL) &key xs)
  (with-slots (x w b) l
    (setf x xs)
    ($xwpb x w b)))

(defmethod backward-propagate ((l AFL) &key d)
  (with-slots (x w dw db) l
    ($gemm x d :c dw :transa T)
    ($copy ($sum d :axis :column) db)
    ($gemm d w :transb T)))

(defclass DUMMYOPT (OPTIMIZER) ())

(defmethod update ((o DUMMYOPT) params gradients)
  (let ((n ($count params)))
    (dotimes (i n)
      (let ((p (elt params i))
            (g (elt params i)))
        ($axpy g p :alpha -0.02)))
    o))

(defun update-parameters (l &key opt)
  (with-slots (w b dw db) l
    (update opt (list w b) (list dw db))))

(defmethod predict ((nw LIST) &key xs)
  (let ((iv xs))
    (dolist (l nw)
      (setf iv (forward-propagate l :xs iv)))
    iv))

(let ((X ($m '((0 0) (0 1) (1 0) (1 1))))
      (y ($m '(0 1 1 1) 1))
      (layers (list ($afl 2 1 :winit :xavier)))
      (v nil)
      (ys nil)
      (dout nil)
      (o (make-instance 'DUMMYOPT)))
  (dotimes (i 20)
    (setf v X)
    (setf ys (predict layers :xs v))
    (setf dout ($/ ($- ys y) ($nrow y)))
    (print ($str i ": " ($mse ys y)))
    (dolist (l (reverse layers))
      (setf dout (backward-propagate l :d dout)))
    (dolist (l layers)
      (update-parameters l :opt o)))
  (setf ys (predict layers :xs v))
  (setf dout ($/ ($- ys y) ($nrow y)))
  (print ($mse ys y))
  (print ys))
