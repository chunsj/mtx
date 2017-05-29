(in-package :mtx)

(declaim (optimize (speed 3) (safety 0) (debug 1)))

(defgeneric $round (m))

(defmethod $round ((n NUMBER)) (* 1.0 (round n)))
(defmethod $round ((m MX)) ($map (lambda (v) (* 1.0 (round v))) m))

(defgeneric $exp (m))

(defmethod $exp ((n NUMBER)) (exp n))
(defmethod $exp ((m MX)) ($map #'exp m))

(defgeneric $log (m))

(defmethod $log ((n NUMBER)) (log n))
(defmethod $log ((m MX)) ($map #'log m))

(defgeneric $log10 (m))

(defmethod $log10 ((n NUMBER)) (log n 10))
(defmethod $log10 ((m MX)) ($map (lambda (v) (log v 10)) m))

(defgeneric $expt (m n))

(defmethod $expt ((m NUMBER) n) (expt m n))
(defmethod $expt ((m MX) n) ($map (lambda (v) (expt v n)) m))

(defgeneric $sqrt (m))

(defmethod $sqrt ((n NUMBER)) (sqrt n))
(defmethod $sqrt ((m MX)) ($map #'sqrt  m))

(defgeneric $sigmoid (m))

(defmethod $sigmoid ((n NUMBER)) (/ 1.0 (+ 1.0 (exp (- n)))))
;;(defmethod $sigmoid ((m MX)) ($map! (lambda (v) (/ 1.0 (+ 1.0 (exp (- v))))) ($dup m)))
;; trying to avoid overflow
(defmethod $sigmoid ((m MX)) ($map! (lambda (v) (* 0.5 (+ 1.0 (tanh (* 0.5 v))))) ($dup m)))

(defgeneric $sigmoid! (m))

(defmethod $sigmoid! ((m MX)) ($map! (lambda (v) (* 0.5 (+ 1.0 (tanh (* 0.5 v))))) m))

(defgeneric $sin (m))

(defmethod $sin ((n NUMBER)) (sin n))
(defmethod $sin ((m MX)) ($map #'sin m))

(defgeneric $cos (m))

(defmethod $cos ((n NUMBER)) (cos n))
(defmethod $cos ((m MX)) ($map #'cos m))

(defgeneric $tan (m))

(defmethod $tan ((n NUMBER)) (tan n))
(defmethod $tan ((m MX)) ($map #'tan m))

(defgeneric $sinh (m))

(defmethod $sinh ((n NUMBER)) (sinh n))
(defmethod $sinh ((m MX)) ($map #'sinh m))

(defgeneric $cosh (m))

(defmethod $cosh ((n NUMBER)) (cosh n))
(defmethod $cosh ((m MX)) ($map #'cosh m))

(defgeneric $tanh (m))

(defmethod $tanh ((n NUMBER)) (tanh n))
(defmethod $tanh ((m MX)) ($map #'tanh m))

(defgeneric $asin (m))

(defmethod $asin ((n NUMBER)) (asin n))
(defmethod $asin ((m MX)) ($map #'asin m))

(defgeneric $acos (m))

(defmethod $acos ((n NUMBER)) (acos n))
(defmethod $acos ((m MX)) ($map #'acos m))

(defgeneric $atan (m))

(defmethod $atan ((n NUMBER)) (atan n))
(defmethod $atan ((m MX)) ($map #'atan m))

(defgeneric $softmax (m &key))

(defmethod $softmax ((n NUMBER) &key) n)
(defmethod $softmax ((m MX) &key (axis :row))
  (let* ((mx ($max m :axis axis))
         (exp-m ($exp ($- m mx)))
         (sum-exp-m ($sum exp-m :axis axis))
         (y ($/ exp-m sum-exp-m)))
    y))

(defgeneric $relu (m))

(defmethod $relu ((n NUMBER)) (max 0 n))
(defmethod $relu ((m MX)) ($map (lambda (v) (max 0 v)) m))

(defgeneric $relu! (m))

(defmethod $relu! ((m MX)) ($map! (lambda (v) (max 0 v)) m))

(defgeneric $lkyrelu (m &key))

(defmethod $lkyrelu ((n NUMBER) &key (lky 0.01)) (if (plusp n) n (* lky n)))
(defmethod $lkyrelu ((m MX) &key (lky 0.01)) ($map (lambda (n) (if (plusp n) n (* lky n))) m))

(defgeneric $lkyrelu! (m &key))
(defmethod $lkyrelu! ((m MX) &key (lky 0.01)) ($map! (lambda (n) (if (plusp n) n (* lky n))) m))

(defun $mse (yv tv)
  ($x (/ 1.0 ($nrow yv)) ($sum ($expt ($- yv tv) 2))))

(defun $cee (yv tv)
  (let ((delta 1.0E-7)
        (batch-size ($nrow yv)))
    ($x (/ -1.0 batch-size) ($sum ($log ($+ delta ($sum ($x yv tv) :axis :row)))))))
