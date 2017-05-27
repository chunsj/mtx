(in-package :mtx)

;; matrix creation
(print ($m 32456 11 11))
(print ($m ($iota 10000) 100 100))

;; random matrix
($r 4)
($r 1 4)
($r 4 1)
($r 4 4)

;; add
($+ 1 2 3 4 5)
($+ 1 2 3 ($m '(1 2)))
($+ ($m '(1 0 0 1) 2)
    ($m '(2 0 0 1) 2)
    ($m '(1 2 2 2) 2))

;; subtract
($- 1)
($- 5 4 3 2 1)
($- ($m '(1 2)))
($- 10 ($m '(1 2 3)))
($- 1 2 3 ($m '(1 2)))
($- ($m '(1 0 0 1) 2)
    ($m '(2 0 0 1) 2)
    ($m '(1 2 3 4) 2))

;; multiply, hadamard
($x 1)
($x ($m '(1 2 3 4) 2)
    ($m '(1 2 3 4) 2))
($x 1 2 3 4 5)
($x 1 2 3 ($m '(1 2)))
($x ($m '(1 0 0 1) 2)
    ($m '(2 0 0 1) 2)
    ($m '(1 2 2 2) 2))

;; multiply, matrix
($* 1 2 3 4 5)
($* 1 2 3 ($m '(1 2)))
($* ($m '(1 0 0 1) 2)
    ($m '(1 0 0 1) 2)
    ($m '(1 2 2 1) 2))

;; divide
($/ 1)
($/ 1 2 3)
($/ ($m '(1 2 3)))
($/ 10 ($m '(1 2 3)))
($/ 1 2 3 ($m '(1 2)))
($/ ($m '(1 1 1 1) 2)
    ($m '(2 3 2 1) 2)
    ($m '(1 2 3 4) 2))

;; gemm style performance comparison
(let ((a ($m 1 100 100))
      (b ($m 2 100 100))
      (c ($m 0 100 100))
      (n 10000))
  (time (dotimes (i n) ($mm a b :c c))))

(let ((a ($m 1 100 100))
      (b ($m 2 100 100))
      (n 10000))
  (time (dotimes (i n) ($* a b))))

;; vector broadcasting
(broadcast-vector ($r 4 1) 4)
(broadcast-vector ($r 1 4) 4)

;; xwbp or ($+ (* x w) b)
(let ((x ($m '(0 0 1 0 0 1 1 1 0 0 1 0 0 1 1 1) 2))
      (W1 ($- ($* 2.0 ($r 2 4)) 1.0))
      (b1 ($r 1 4)))
  ($+ ($* x W1) (broadcast-vector b1 ($nrow x))))

(let ((x ($m '(0 0 1 0 0 1 1 1 0 0 1 0 0 1 1 1) 2))
      (W1 ($- ($* 2.0 ($r 2 4)) 1.0))
      (b1 ($r 1 4)))
  ($xwpb X W1 b1))

;; basic neural network computation performance
(let ((X ($r 10000 (* 28 28)))
      (W1 ($r (* 28 28) 100))
      (b1 ($r 1 100))
      (W2 ($r 100 10))
      (b2 ($r 1 10)))
  (time
   (dotimes (i 100)
     (let* ((Z1 ($xwpb X W1 b1))
            (A1 ($sigmoid Z1))
            (Z2 ($xwpb A1 W2 b2))
            (A2 ($sigmoid Z2)))
       A2))))

;; slower than nd4j case
(let* ((n 10000)
       (X ($r n (* 28 28)))
       (W1 ($r (* 28 28) 100))
       (W2 ($r 100 10))
       (c1 ($m 0 n 100))
       (c2 ($m 0 n 10)))
  (time
   (dotimes (i 100)
     (let* ((Z1 ($mm X W1 :c c1))
            (A1 Z1)
            (Z2 ($mm A1 W2 :c c2))
            (A2 Z2))
       A2))))

;; batch performance comparison, yes, batch is faster
;; 1. single record, 10000 times
(let* ((n 1)
       (X ($r n (* 28 28)))
       (W1 ($r (* 28 28) 100))
       (W2 ($r 100 10))
       (c1 ($m 0 n 100))
       (c2 ($m 0 n 10)))
  (time
   (dotimes (i 10000)
     (let* ((Z1 ($mm X W1 :c c1))
            (A1 Z1)
            (Z2 ($mm A1 W2 :c c2))
            (A2 Z2))
       A2))))
;; 2. 100 records, 100 times
(let* ((n 1000)
       (X ($r n (* 28 28)))
       (W1 ($r (* 28 28) 100))
       (W2 ($r 100 10))
       (c1 ($m 0 n 100))
       (c2 ($m 0 n 10)))
  (time
   (dotimes (i 100)
     (let* ((Z1 ($mm X W1 :c c1))
            (A1 Z1)
            (Z2 ($mm A1 W2 :c c2))
            (A2 Z2))
       A2))))
;; 3. 1000 records, 10 times
(let* ((n 1000)
       (X ($r n (* 28 28)))
       (W1 ($r (* 28 28) 100))
       (W2 ($r 100 10))
       (c1 ($m 0 n 100))
       (c2 ($m 0 n 10)))
  (time
   (dotimes (i 10)
     (let* ((Z1 ($mm X W1 :c c1))
            (A1 Z1)
            (Z2 ($mm A1 W2 :c c2))
            (A2 Z2))
       A2))))
;; 4. 10000 records, 1 times
(let* ((n 10000)
       (X ($r n (* 28 28)))
       (W1 ($r (* 28 28) 100))
       (W2 ($r 100 10))
       (c1 ($m 0 n 100))
       (c2 ($m 0 n 10)))
  (time
   (dotimes (i 1)
     (let* ((Z1 ($mm X W1 :c c1))
            (A1 Z1)
            (Z2 ($mm A1 W2 :c c2))
            (A2 Z2))
       A2))))

;; with thread macro
(let* ((n 10000)
       (X ($r n (* 28 28)))
       (W1 ($r (* 28 28) 100))
       (b1 ($r 1 100))
       (W2 ($r 100 10))
       (b2 ($r 1 10)))
  (@> X
      ($xwpb W1 b1)
      ($sigmoid)
      ($xwpb W2 b2)
      ($sigmoid)))

;; elementwise utility functions
(let* ((a ($r 4 4)))
  (print a)
  (print ($max a))
  (print ($imax a))
  (print ($min a))
  (print ($imin a)))

;; indices of max element by axis
(let ((a ($r 5 4)))
  (print a)
  (print ($argmax a))
  (print ($argmax a :axis :column)))

;; more basic performance test codes
;; allocation comparison
(time (dotimes (i 1000) ($vx (* 100 100))))
(time (dotimes (i 1000) ($m 0 100 100)))
(time (dotimes (i 1000) ($dup ($m 0 100 100))))
(let ((r ($r 100 100)))
  (time (dotimes (i 1000) ($copy ($m 0 100 100) r))))
(let ((r ($r 100 100))
      (q ($r 100 100)))
  (time (dotimes (i 1000) ($copy q r))))
;; transpose uses allocation
(let ((r ($r 100 100)))
  (time (dotimes (i 1000) ($transpose r))))

;; XXX transpose should be processed with view; realize it when it is required
;; XXX here some time can be reduced
(let ((r ($r 100 100)))
  (time (dotimes (i 1000) ($mm r ($transpose r)))))

(let ((r ($r 100 100)))
  (time (dotimes (i 1000) ($gemm r r :transb T))))

(let ((r ($r 100 100))
      (c ($m 0 100 100)))
  (time (dotimes (i 1000) ($gemm r r :c c :transb T))))

;; multiplication with allocated result
(let ((a ($m 1.0 1 10000))
      (b ($m 2.0 10000 1))
      (c ($m 0 1 1)))
  (time (dotimes (i 1000) ($mm a b :c c)))
  c)

;; direct blas method comparison, double vs single float
(let ((a (make-fnv-double 10000 :initial-value 1))
      (b (make-fnv-double 10000 :initial-value 2))
      (c (make-fnv-double 10000 :initial-value 0)))
  (time (dotimes (i 1000) (%dgemm "N" "N" 100 100 100 1.0 a 100 b 100 0.0 c 100))))

(let ((a (make-fnv-float 10000 :initial-value 1))
      (b (make-fnv-float 10000 :initial-value 2))
      (c (make-fnv-float 10000 :initial-value 0)))
  (time (dotimes (i 1000) (%sgemm "N" "N" 100 100 100 1.0 a 100 b 100 0.0 c 100))))
