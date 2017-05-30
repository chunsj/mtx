(in-package :mtx)

(let ((X ($r 10000 (* 28 28)))
      (W1 ($r (* 28 28) 100))
      (W2 ($r 100 10)))
  (time
   (dotimes (n 100)
     (let* ((Z1 ($* X W1))
            (A1 Z1)
            (Z2 ($* A1 W2))
            (A2 Z2))
       A2))))
