(in-package :mtx)

(let* ((m ($m '((1 2 3 0)
                (0 1 2 3)
                (3 0 1 2)
                (2 3 0 1))))
       (f ($m '((2 0 1)
                (0 1 2)
                (1 0 2)))))
  ($convolute m f 3))

(defparameter *mnist* (read-mnist-data))

(let* ((train-images (getf *mnist* :train-images)))
  ($reshape ($ train-images 0 T) 28 28))
