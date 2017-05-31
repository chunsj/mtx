(in-package :mtx)


(let* ((m ($m '(0 1 2 3 4 5 6 7 8 9 10 11)))
       (f ($m '(1 0 1 2 1 2 3 2 3))))
  ($convolute m 4 4 f 2 :b 0.0))

(defparameter *mnist* (read-mnist-data))

(let* ((m ($m '((1 2 3 0)
                (0 1 2 3)
                (3 0 1 2)
                (2 3 0 1))))
       (pm ($ptr m))
       (mh ($nrow m))
       (mw ($ncol m))
       (f ($m '((2 0 1)
                (0 1 2)
                (1 0 2))))
       (pf ($ptr f))
       (fw ($ncol f))
       (fh ($nrow f))
       (stride 1)
       (padding 0)
       (ch (1+ (/ (+ (- mh fh) (* 2 padding)) stride)))
       (cw (1+ (/ (+ (- mw fw) (* 2 padding)) stride)))
       (cm ($m 0 1 (* ch cw fh fw)))
       (pcm ($ptr cm))
       (n 0))
  (list pm pf)
  (time
   (dotimes (kkk 1000)
     (setf n 0)
     (loop :for i :from 0 :by stride :below ch :do
        (loop :for j :from 0 :by stride :below cw :do
           (let* ((sm ($sm m i j fh fw))
                  (psm ($ptr sm))
                  (l (* n (* fh fw))))
             ;;(print ($fnv sm))
             (dotimes (k (* fh fw))
               (setf ($prf pcm (+ k l)) ($prf psm k)))
             ;;(print ($sm cm 0 l 1 (* fh fw)))
             (incf n))))))
  cm)


(defun $mkcl (m nr nc fs &key (padding 0) (stride 1))
  ;; m and f should be in the form of row vector
  (let ((cr (1+ (/ (+ (- nr fs) (* 2 padding)) stride)))
        (cc (1+ (/ (+ (- nc fs) (* 2 padding)) stride))))
    (assert (and (integerp cr) (integerp cc))
            nil
            "dimension mismatched")
    (when (and (integerp cr) (integerp cc))
      (let* ((mr (+ nr (* 2 padding)))
             (mc (+ nc (* 2 padding)))
             (fd (1- fs))
             (cl ($zeros 1 (* fs fs cr cc)))
             (cli 0))
        (loop :for i :from 0 :below mr :by stride :while (< (+ i fd) mr) :do
           (loop :for j :from 0 :below mc :by stride :while (< (+ j fd) mc) :do
              (loop :for ik :from 0 :below fs :do
                 (loop :for jk :from 0 :below fs :do
                    (let ((mv (vofm m nr nc padding (+ i ik) (+ j jk))))
                      (setf ($ cl 0 cli) mv)
                      (incf cli))))))
        cl))))

(defun im2cl (m fh fw stride padding)
  (let* ((mh ($nrow m))
         (mw ($ncol m))
         (ch (1+ (/ (+ (- mh fh) (* 2 padding)) stride)))
         (cw (1+ (/ (+ (- mw fw) (* 2 padding)) stride)))
         (cm ($zeros 1 (* ch cw fh fw)))
         (pcm ($ptr cm)))
    (loop :for i :from padding :by stride :below ch :do
       (loop :for j :from padding :by stride :below cw :do
          (let* ((sm ($sm m i j fh fw T))
                 (psm ($ptr sm))
                 (l (+ (* i (* cw fh fw)) (* j (* fh fw)))))
            (dotimes (k (* fh fw))
              (setf ($prf pcm (+ k l)) ($prf psm k))))))
    cm))

(let ((m ($m '((1 2 3 0)
               (0 1 2 3)
               (3 0 1 2)
               (2 3 0 1)))))
  ($mkcl ($rowv m) 4 4 3))

(let ((m ($reshape ($m '((1 2 3 0)
                         (0 1 2 3)
                         (3 0 1 2)
                         (2 3 0 1)))
                   1 16)))
  ($sm m 0 0 3 3 T))

(let* ((m ($m '((1 2 3 0)
                (0 1 2 3)
                (3 0 1 2)
                (2 3 0 1))))
       (f ($m '((2 0 1)
                (0 1 2)
                (1 0 2)))))
  ($convolute ($reshape m 1 16) 4 4 ($reshape f 1 9) 3))

(let* ((m ($m '((1 2 3 0)
                (0 1 2 3)
                (3 0 1 2)
                (2 3 0 1))))
       (f ($m '((2 0 1)
                (0 1 2)
                (1 0 2)))))
  (im2cl m ($nrow f) ($ncol f) 1 0))

(let* ((m ($m '((1 2 3 0)
                (0 1 2 3)
                (3 0 1 2)
                (2 3 0 1))))
       (f ($m '((2 0 1)
                (0 1 2)
                (1 0 2)))))
  (time (dotimes (i 1000) (im2cl m ($nrow f) ($ncol f) 1 0))))

(let* ((train-images (getf *mnist* :train-images)))
  (im2cl ($reshape ($ train-images 0 T) 28 28) 3 3 1 0))

(let* ((train-images (getf *mnist* :train-images)))
  (time
   (dotimes (n 1000)
     (im2cl ($reshape ($ train-images n T) 28 28) 3 3 1 0))))

(let* ((train-images (getf *mnist* :train-images)))
  (let ((f ($m '((2 0 1) (0 1 2) (1 0 2))))
        (r ($zeros 1 676)))
    (time
     (dotimes (n 1000)
       (let ((mcl (im2cl ($reshape ($ train-images n T) 28 28) 3 3 1 0)))
         ($gemm ($reshape f 1 9)
                ($reshape mcl 9 (/ ($size mcl) 9))
                :c r))))))


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

(defun conv-at (images w h i)
  (let ((m ($reshape ($ images i T) w h))
        (f ($m '((2 0 1)
                 (0 1 2)
                 (1 0 2)))))
    ($convolute m f :stride 1)))

(let* ((train-images (getf *mnist* :train-images))
       (f ($m '(2 0 1 0 1 2 1 0 2)))
       (X ($ train-images 0 T))
       (c1 ($convolute X 28 28 f 3 :padding 1 :stride 3)))
  c1)

(let* ((train-images (getf *mnist* :train-images))
       (X ($ train-images 0 T)))
  (time (dotimes (i 10) ($mkcl X 28 28 3))))

(* 26 26 9)

(let* ((train-images (getf *mnist* :train-images))
       (f ($m '(2 0 1 0 1 2 1 0 2)))
       (ntr 1000))
  (time (dotimes (i ntr) ($convolute ($ train-images i T) 28 28 f 3))))

($reshape ($m '((2 0 1) (0 1 2) (1 0 2))) 1 9)

(let* ((train-images (getf *mnist* :train-images))
       (c1 (conv-at train-images 28 28 0))
       (f ($m '((2 0 1) (0 1 2) (1 0 2))))
       (c2 ($convolute ($reshape c1 26 26) f))
       (c3 ($convolute ($reshape c2 24 24) f))
       (c4 ($convolute ($reshape c3 22 22) f))
       (c5 ($convolute ($reshape c4 20 20) f)))
  c5)

;; XXX too slow yet
(let* ((train-images (getf *mnist* :train-images)))
  (time (dotimes (i 1000) (conv-at train-images 28 28 i))))
