(in-package :mtx)

(defclass SIGMOIDLAYER (LAYER)
  ((out :initform nil)))

(defun $sigmoid-layer () (make-instance 'SIGMOIDLAYER))

(defmethod forward-propagate ((l SIGMOIDLAYER) &key xs train)
  (declare (ignore train))
  (with-slots (out) l
    (setf out ($sigmoid xs))
    out))

(defmethod backward-propagate ((l SIGMOIDLAYER) &key d)
  (with-slots (out) l
    ($x d ($- 1.0 out) out)))

(defclass TANHLAYER (LAYER)
  ((out :initform nil)))

(defun $tanh-layer () (make-instance 'TANHLAYER))

(defmethod forward-propagate ((l TANHLAYER) &key xs train)
  (declare (ignore train))
  (with-slots (out) l
    (setf out ($tanh xs))
    out))

(defmethod backward-propagate ((l TANHLAYER) &key d)
  (with-slots (out) l
    ($x d ($- 1.0 ($x out out)))))

(defclass RELULAYER (LAYER)
  ((out :initform nil)))

(defun $relu-layer () (make-instance 'RELULAYER))

(defmethod forward-propagate ((l RELULAYER) &key xs train)
  (declare (ignore train))
  (with-slots (out) l
    (setf out ($relu xs))
    out))

(defmethod backward-propagate ((l RELULAYER) &key d)
  (with-slots (out) l
    ($map (lambda (de oe) (if (> oe 0) de 0.0)) d out)))

(defclass LKYRELULAYER (LAYER)
  ((lky :initform 0.01 :initarg :l)
   (out :initform nil)))

(defun $lkyrelu-layer (&key (lky 0.01)) (make-instance 'LKYRELULAYER :l lky))

(defmethod forward-propagate ((l LKYRELULAYER) &key xs train)
  (declare (ignore train))
  (with-slots (out lky) l
    (setf out ($lkyrelu xs :lky lky))
    out))

(defmethod backward-propagate ((l LKYRELULAYER) &key d)
  (with-slots (out lky) l
    ($map (lambda (de oe) (if (> oe 0) de (* lky de))) d out)))

(defclass SOFTMAXCEELOSSLAYER (LAYER)
  ((loss :initform nil)
   (ys :reader ys :initform nil)
   (ts :reader ts :initform nil)))

(defun $softmax-layer () (make-instance 'SOFTMAXCEELOSSLAYER))

(defmethod forward-propagate ((l SOFTMAXCEELOSSLAYER) &key ys ts)
  (with-slots (loss (lts ts) (lys ys)) l
    (setf lts ts)
    (setf lys ($softmax ys :axis :row))
    (setf loss ($cee ys lts))
    loss))

(defmethod backward-propagate ((l SOFTMAXCEELOSSLAYER) &key)
  (with-slots (ts ys) l
    ($/ ($- ys ts) ($nrow ts))))

(defclass PLAINMSELOSSLAYER (LAYER)
  ((loss :initform nil)
   (ys :reader ys :initform nil)
   (ts :reader ts :initform nil)))

(defun $mse-layer () (make-instance 'PLAINMSELOSSLAYER))

(defmethod forward-propagate ((l PLAINMSELOSSLAYER) &key ys ts)
  (with-slots (loss (lts ts) (lys ys)) l
    (setf lts ts)
    (setf lys ys)
    (setf loss ($mse lys lts))
    loss))

(defmethod backward-propagate ((l PLAINMSELOSSLAYER) &key)
  (with-slots (ts ys) l
    ($/ ($- ys ts) ($nrow ts))))

(defclass DROPOUTLAYER (LAYER)
  ((dr :initarg :dr :initform nil)
   (mask :initform nil)))

(defun $dropout-layer (&key (dr 0.5)) (make-instance 'DROPOUTLAYER :dr dr))

(defmethod forward-propagate ((l DROPOUTLAYER) &key xs (train T))
  (with-slots (dr mask) l
    (when train
      (setf mask ($map (lambda (v) (if (> v dr) 1.0 0.0))
                       ($r ($nrow xs) ($ncol xs)))))
    (if train
        ($x xs mask)
        ($x xs (- 1.0 dr)))))

(defmethod backward-propagate ((l DROPOUTLAYER) &key d)
  (with-slots (mask) l
    (if mask
        ($x d mask)
        d)))

(defclass AFFINELAYER (LAYER)
  ((input-size :initarg :input-size :initform nil)
   (output-size :initarg :output-size :initform nil)
   (wdl :initarg :wdl :initform 0.0)
   (w :initarg nil)
   (b :initarg nil)
   (x :initform nil)
   (dw :reader dw :initform nil)
   (db :reader db :initform nil)))

(defmethod print-object ((l AFFINELAYER) stream)
  (print-unreadable-object (l stream :type t :identity t)
    (with-slots (input-size output-size wdl) l
      (format stream "[~A x ~A] WITH ~10,2E" input-size output-size wdl))))

(defun $affine-layer (input-size output-size &key (winit :he) (wdl 0.0))
  (let ((l (make-instance 'AFFINELAYER :input-size input-size :output-size output-size
                          :wdl wdl)))
    (with-slots (w b dw db) l
      (let ((sc (cond ((eq winit :he) (sqrt (/ 6.0 input-size)))
                      ((eq winit :xavier) (sqrt (/ 3.0 input-size)))
                      (T 1.0))))
        (setf w ($x sc ($- ($x 2.0 ($r input-size output-size)) 1.0))
              dw ($m 0.0 input-size output-size))
        (setf b ($m 1.0 1 output-size)
              db ($m 0.0 1 output-size)))
      l)))

(defmethod forward-propagate ((l AFFINELAYER) &key xs train)
  (declare (ignore train))
  (with-slots (x w b) l
    (setf x xs)
    ($xwpb x w b)))

(defmethod backward-propagate ((l AFFINELAYER) &key d)
  (with-slots (wdl x w dw db) l
    ($gemm x d :c dw :transa T)
    (when (> (abs wdl) 0) ($axpy dw dw :alpha wdl))
    ($copy ($sum d :axis :column) db)
    ($gemm d w :transb T)))

(defmethod optimize-parameters ((l AFFINELAYER) &key o)
  (with-slots (w b dw db) l
    (update o (list w b) (list dw db))))

(defmethod regularization ((l AFFINELAYER))
  (with-slots (wdl w) l
    (* 0.5 wdl ($sum ($x w w)))))

(defmethod parameters ((l AFFINELAYER))
  (with-slots (w b) l (list w b)))

;; XXX need to be fixed to new architecture
(defclass BATCHNORMLAYER (LAYER)
  ((gamma :initform nil)
   (beta :initform nil)
   (momentum :initarg :m :initform nil)
   (rmean :initarg :rm :initform nil)
   (rvar :initarg :rv :initform nil)
   (batch-size :initform nil)
   (xc :initform nil)
   (xn :initform nil)
   (std :initform nil)
   (dgamma :initform nil :reader dgamma)
   (dbeta :initform nil :reader dbeta)))

(defun $batchnorm-layer (sz &key (m 0.9) rm rv)
  (let ((l (make-instance 'BATCHNORMLAYER :m m :rm rm :rv rv)))
    (with-slots (gamma beta) l
      (setf gamma ($ones 1 sz))
      (setf beta ($zeros 1 sz)))
    l))

(defun bn-forward-h (l x tf)
  (with-slots (rmean rvar batch-size xc xn std momentum beta gamma) l
    (when (null rmean)
      (setf rmean ($zeros 1 ($ncol x)))
      (setf rvar ($zeros 1 ($ncol x))))
    (if tf
        (let* ((tmu ($mean x :axis :column))
               (txc ($- x tmu))
               (tvar ($mean ($x x x) :axis :column))
               (tstd ($sqrt ($+ tvar 10E-7)))
               (txn ($/ txc tstd)))
          (setf batch-size ($nrow x))
          (setf xc txc)
          (setf xn txn)
          (setf std tstd)
          (setf rmean ($+ ($x momentum rmean) ($x (- 1.0 momentum) tmu)))
          (setf rvar ($+ ($x momentum rvar) ($x (- 1.0 momentum) tvar))))
        (progn
          (setf xc ($- x rmean))
          (setf xn ($/ xc ($sqrt ($+ rvar 10E-7))))))
    ($+ ($x gamma xn) beta)))

(defmethod forward-propagate ((l BATCHNORMLAYER) &key xs (train T))
  (bn-forward-h l xs train))

(defun bn-backward-h (l dout)
  (with-slots (gamma dbeta dgamma std xn xc batch-size) l
    (let* ((tdbeta ($sum dout :axis :column))
           (tdgamma ($sum ($x xn dout) :axis :column))
           (dxn ($x gamma dout))
           (dxc ($/ dxn std))
           (dstd ($- ($sum ($/ ($x dxn xc) ($x std std)) :axis :column)))
           (dvar ($/ ($x 0.5 dstd) std))
           (dxc ($axpy ($x (/ 2.0 batch-size) xc dvar) dxc))
           (dmu ($sum dxc :axis :column))
           (dx ($- dxc ($/ dmu batch-size))))
      (setf dgamma tdgamma)
      (setf dbeta tdbeta)
      dx)))

(defmethod backward-propagate ((l BATCHNORMLAYER) &key d)
  (bn-backward-h l d))

(defmethod optimize-parameters ((l BATCHNORMLAYER) &key o)
  (with-slots (gamma beta dgamma dbeta) l
    (update o (list gamma beta) (list dgamma dbeta))))

(defmethod parameters ((l BATCHNORMLAYER))
  (with-slots (gamma beta) l (list gamma beta)))
