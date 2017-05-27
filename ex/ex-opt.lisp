(in-package :mtx)

(defun f (x y) ($+ (/ ($expt x 2.0) 20.0) ($expt y 2.0)))
(defun df (x y) (list ($/ x 10.0) ($* 2.0 y)))

(let* ((init-pos (list ($m '(-7.0)) ($m '(2.0))))
       (params init-pos)
       (grads (list ($m '(0)) ($m '(0))))
       (optimizers (list :SGD (make-instance 'SGD :lr 0.95)
                         :MOMENTUM (make-instance 'MOMENTUM :lr 0.1 :m 0.9)
                         :ADAGRAD (make-instance 'ADAGRAD :lr 1.5)
                         :ADAM (make-instance 'ADAM :lr 0.3 :b1 0.9 :b2 0.999))))
  (dolist (k (loop :for c :in optimizers :by #'cddr :collect c))
    (let ((optimizer (getf optimizers k))
          (param-history nil))
      (setf params init-pos)
      (dotimes (n 30)
        (push params param-history)
        (setf grads (df (car params) (cadr params)))
        (update optimizer params grads))
      (print ($str k " " (car param-history))))))