(in-package :mtx)

;; https://iamtrask.github.io/2017/03/21/synthetic-gradients/

(defun binary-list (n &optional acc)
  (cond ((zerop n) (or acc (list 0)))
        ((plusp n)
         (binary-list (ash n -1) (cons (logand 1 n) acc)))
        (t (error "~S: non-negative argument required, got ~s" 'binary-list n))))

(defun zeros-list (n) (loop :for i :from 0 :below n :collect 0))

(defun ->binary (n sz)
  (let ((l0 (binary-list n)))
    (append (zeros-list (- sz ($count l0))) l0)))

(defun ->decimal (l)
  (let ((n ($count l)))
    (loop :for i :from 0 :below n
       :sum (* (elt l i) (expt 2 (- n (1+ i)))))))

(defun generate-dataset (&key (dim 8) (num 1000))
  (let* ((sample-size num)
         (num-dim dim)
         (bn (expt 2 (1- num-dim)))
         (X ($m 0 sample-size (* 2 num-dim)))
         (y ($m 0 sample-size num-dim)))
    (loop :for i :from 0 :below sample-size :do
       (let* ((x0int (random bn))
              (x1int (random bn))
              (yint (+ x0int x1int)))
         (setf ($ X i T) ($m (append (->binary x0int num-dim) (->binary x1int num-dim))))
         (setf ($ y i T) ($m (->binary yint num-dim)))))
    (list :x X :y y)))

(let* ((num-samples 100)
       (output-dim 8)
       (dataset (generate-dataset :dim output-dim :num num-samples))
       (X (getf dataset :x))
       (y (getf dataset :y))
       (o ($adagrad-optimizer :lr 0.08))
       (layers (list ($affine-layer (* 2 output-dim) 128 :winit :xavier)
                     ($sigmoid-layer)
                     ($affine-layer 128 64)
                     ($sigmoid-layer)
                     ($affine-layer 64 output-dim)))
       (n ($snn layers :el ($mse-layer) :o o))
       (ntr 2000))
  (print ($str "INITIAL: " ($round (predict n :xs X))))
  (time (dotimes (i ntr) (train n :xs X :ts y)))
  (print ($str "FINAL: " ($round (predict n :xs X))))
  (print ($str "TRUE: " ($round y)))
  (print ($str "LOSS: " ($mse ($round (predict n :xs X)) y)))
  (print ($str "LSST: " ($mse y y))))
