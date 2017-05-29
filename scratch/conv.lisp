(in-package :mtx)

(let* ((m ($m '((1 2 3 0)
                (0 1 2 3)
                (3 0 1 2)
                (2 3 0 1))))
       (f ($m '((2 0 1)
                (0 1 2)
                (1 0 2)))))
  ($convolute m f))

(let* ((m ($m '((1 2 3 0)
                (0 1 2 3)
                (3 0 1 2)
                (2 3 0 1))))
       (f ($m '((2 0 1)
                (0 1 2)
                (1 0 2)))))
  ($convolute m f :padding 1 :stride 3))

(defparameter *mnist* (read-mnist-data))

(defun conv-at (images i)
  (let ((m ($reshape ($ images i T) 28 28))
        (f ($m '((2 0 1)
                 (0 1 2)
                 (1 0 2)))))
    ($convolute m f :stride 3)))

;; XXX too slow
(let* ((train-images (getf *mnist* :train-images)))
  (time (dotimes (i ($nrow train-images)) (conv-at train-images i))))
