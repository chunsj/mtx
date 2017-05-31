(in-package :mtx)

(declaim (optimize (speed 3) (safety 0) (debug 1)))

(defun $vx (n &key (initial-value 0.0))
  (make-fnv-float n :initial-value initial-value))

(defgeneric $count (s))

(defmethod $count ((s LIST)) (length s))
(defmethod $count ((s ARRAY)) (length s))
(defmethod $count ((s FNV-FLOAT)) (fnv-length s))

(defmacro $ref (v i) `(fnv-float-ref ,v ,i))
(defmacro $prf (p i) `(fnv-float-ptr-ref ,p ,i))

(defclass MX ()
  ((fnv :initarg :fnv :accessor $fnv :initform nil)
   (nrows :initarg :nrow :accessor $nrow :initform 0)
   (ncols :initarg :ncol :accessor $ncol :initform 0)))

(defgeneric $ptr (m))
(defmethod $ptr ((v FNV-FLOAT)) (fnv-float-foreign-pointer v))
(defmethod $ptr ((m MX)) (fnv-float-foreign-pointer ($fnv m)))

(defgeneric $size (m))

(defmethod $size ((n NUMBER)) 1)
(defmethod $size ((v FNV-FLOAT)) ($count v))
(defmethod $size ((m MX)) ($count ($fnv m)))

(defgeneric $dim (m))

(defmethod $dim ((n NUMBER)) nil)
(defmethod $dim ((v FNV-FLOAT)) (list ($size v)))
(defmethod $dim ((m MX)) (list ($nrow m) ($ncol m)))

(defun $matrix? (m)
  (and (typep m 'MX) (and (/= 1 ($nrow m)) (/= 1 ($ncol m)))))

(defun $vector? (m)
  (and (typep m 'MX) (or (= 1 ($nrow m)) (= 1 ($ncol m)))))

(defun $row-vector? (m)
  (and (typep m 'MX) (= 1 ($nrow m))))

(defun $column-vector? (m)
  (and (typep m 'MX) (= 1 ($ncol m))))

(defun $scalar? (m)
  (and (= 1 ($nrow m)) (= 1 ($ncol m))))

(defun ->scalar (m)
  (if ($scalar? m)
      ($ref ($fnv m) 0)
      m))

(defun $reshape (m nrow ncol)
  (let ((nm ($zeros nrow ncol))
        (tm ($fnv ($transpose m))))
    (dotimes (i nrow)
      (dotimes (j ncol)
        (setf ($ nm i j) ($ref tm (+ (* i ncol) j)))))
    nm))

(defmethod print-object ((m MX) stream)
  (let* ((nr0 ($nrow m))
         (nc0 ($ncol m))
         (nr nr0)
         (nc nc0)
         (col-truncated? nil)
         (row-truncated? nil)
         (maxn 8)
         (halfn (/ maxn 2)))
    (when (> nr maxn)
      (setf row-truncated? T)
      (setf nr maxn))
    (when (> nc maxn)
      (setf col-truncated? T)
      (setf nc maxn))
    (format stream "MX[~A x ~A] : ~%" nr0 nc0)
    (loop :for i :from 0 :below nr
       :do (let ((i (if (and row-truncated? (>= i halfn)) (- nr0 (- maxn i)) i)))
             (format stream "")
             (loop :for j :from 0 :below nc
                :do (let ((j (if (and col-truncated? (>= j halfn)) (- nc0 (- maxn j)) j)))
                      (if (< j (1- nc0))
                          (if (and col-truncated? (= j (- nc0 halfn)))
                              (format stream " ··· ~10,2E " ($ref ($fnv m) (+ i (* nr0 j))))
                              (format stream "~10,2E " ($ref ($fnv m) (+ i (* nr0 j)))))
                          (format stream "~10,2E" ($ref ($fnv m) (+ i (* nr0 j)))))))
             (if (and row-truncated? (= i (1- halfn)))
                 (format stream "~%~%   ···~%~%")
                 (format stream "~%"))))))

(declaim (inline make-vx-from-seq))
(defun make-vx-from-seq (s r c)
  (let* ((v ($vx (length s)))
         (pv ($ptr v)))
    (dotimes (j c)
      (dotimes (i r)
        (setf ($prf pv (+ (* r j) i)) (* 1.0 (elt s (+ (* c i) j))))))
    v))

(declaim (inline $mx))
(defun $mx (values &key (nrow nil) (ncol nil))
  (cond ((typep values 'NUMBER) (make-instance 'MX :fnv ($vx (* nrow ncol) :initial-value values)
                                               :nrow 1
                                               :ncol 1))
        ((or (typep values 'ARRAY)
             (typep values 'LIST)) (make-instance 'MX :fnv (make-vx-from-seq values nrow ncol)
                                                  :nrow nrow
                                                  :ncol ncol))
        ((typep values 'FNV-FLOAT) (make-instance 'MX :fnv values
                                                  :nrow nrow
                                                  :ncol ncol))))

(declaim (inline build-mx))
(defun build-mx (seq)
  (let ((e (elt seq 0)))
    (cond ((or (typep e 'LIST) (typep e 'ARRAY)) (let* ((nr (length seq))
                                                        (nc (length e))
                                                        (vs ($vx (* nr nc)))
                                                        (vp ($ptr vs)))
                                                   (dotimes (j nc)
                                                     (dotimes (i nr)
                                                       (setf ($prf vp (+ (* nr j) i))
                                                             (* 1.0 (elt (elt seq i) j)))))
                                                   ($mx vs :nrow nr :ncol nc)))
          ((typep e 'FNV-FLOAT) (let* ((nr (length seq))
                                       (nc ($count e))
                                       (vs ($vx (* nr nc)))
                                       (vp ($ptr vs)))
                                  (dotimes (j nc)
                                    (dotimes (i nr)
                                      (setf ($prf vp (+ (* nr j) i)) ($ref (elt seq i) j))))
                                  ($mx vs :nrow nr :ncol nc)))
          ((typep e 'NUMBER) ($mx seq :nrow 1 :ncol ($count seq))))))

(defun $m (data &optional (d0 nil) (d1 nil))
  (cond ((typep data 'MX) data)
        ((and (typep data 'NUMBER) d0 d1) ($mx ($vx (* d0 d1) :initial-value (* 1.0 data))
                                               :nrow d0 :ncol d1))
        ((typep data 'NUMBER) ($mx ($vx 1 :initial-value data) :nrow 1 :ncol 1))
        ((and (or (typep data 'LIST) (typep data 'ARRAY)) (eq d0 nil) (eq d1 nil)) (build-mx data))
        ((and (or (typep data 'LIST) (typep data 'ARRAY)) d0 d1) ($mx data :nrow d0 :ncol d1))
        ((and (typep data 'FNV-FLOAT) d0) ($mx data :nrow (/ ($count data) d0) :ncol d0))
        ((and (or (typep data 'LIST) (typep data 'ARRAY)) d0) ($mx data
                                                                   :nrow (/ ($count data) d0)
                                                                   :ncol d0))))

(defun $rvx (n)
  (let ((vs ($vx n)))
    (loop :for i :from 0 :below n :do (setf ($ref vs i) (random 1.0)))
    vs))

(defun randn ()
  (* (sqrt (* -2.0 (log (- 1.0 (random 1.0)))))
     (cos (* 2.0 PI (random 1.0)))))

(defun $rnvx (n)
  (let ((vs ($vx n)))
    (loop :for i :from 0 :below n :do (setf ($ref vs i) (randn)))
    vs))

(defun $r (&optional d0 d1)
  (cond ((and (eq d0 nil) (eq d1 nil)) (random 1.0))
        ((and d0 (eq d1 nil)) ($mx ($rvx d0) :nrow 1 :ncol d0))
        ((and d0 d1) ($mx ($rvx (* d0 d1)) :nrow d0 :ncol d1))))

(defun $rn (&optional d0 d1)
  (cond ((and (eq d0 nil) (eq d1 nil)) (randn))
        ((and d0 (eq d1 nil)) ($mx ($rnvx d0) :nrow 1 :ncol d0))
        ((and d0 d1) ($mx ($rnvx (* d0 d1)) :nrow d0 :ncol d1))))

(defun $ones (&optional d0 d1)
  (cond ((and (eq d0 nil) (eq d1 nil)) 1.0)
        ((and d0 (eq d1 nil)) ($mx (loop :for i :from 0 :below d0 :collect 1.0) :nrow 1 :ncol d0))
        ((and d0 d1) ($mx (loop :for i :from 0 :below (* d0 d1) :collect 1.0) :nrow d0 :ncol d1))))

(defun $zeros (&optional d0 d1)
  (cond ((and (eq d0 nil) (eq d1 nil)) 0.0)
        ((and d0 (eq d1 nil)) ($mx (loop :for i :from 0 :below d0 :collect 0.0) :nrow 1 :ncol d0))
        ((and d0 d1) ($mx (loop :for i :from 0 :below (* d0 d1) :collect 0.0) :nrow d0 :ncol d1))))

(defun $ (m &optional (i T) (j T))
  (cond ((not (typep m 'mx)) nil)
        ((and (eq i T) (eq j T)) m)
        ((and (eq i T) (numberp j)) (let* ((nr ($nrow m))
                                           (r ($m 0 nr 1))
                                           (mvs ($ptr m))
                                           (rvs ($ptr m)))
                                      (dotimes (ii nr)
                                        (setf ($prf rvs ii) ($prf mvs (+ (* nr j) ii))))
                                      r))
        ((and (numberp i) (eq j T)) (let* ((nc ($ncol m))
                                           (nr ($nrow m))
                                           (r ($m 0 1 nc))
                                           (mvs ($ptr m))
                                           (rvs ($ptr r)))
                                      (dotimes (jj nc)
                                        (setf ($prf rvs jj) ($prf mvs (+ i (* nr jj)))))
                                      r))
        ((and (numberp i) (numberp j)) (let* ((nr ($nrow m))
                                              (mvs ($ptr m)))
                                         ($prf mvs (+ i (* nr j)))))))

(defun $setv (m nv &optional (i T) (j T))
  (cond ((and (eq i T) (eq j T)) (cond ((numberp nv) (let ((sz ($size m))
                                                           (vs ($ptr m)))
                                                       (dotimes (i sz) (setf ($prf vs i) (* 1.0 nv)))
                                                       m))
                                       ((and (typep nv 'MX)
                                             (= ($nrow m) ($nrow nv))
                                             (= ($ncol m) ($ncol nv)))
                                        ($copy nv m)
                                        m)))
        ((and (eq i T) (numberp j)) (cond ((numberp nv)
                                           (let ((mvs ($ptr m))
                                                 (nrm ($nrow m)))
                                             (dotimes (ii nrm)
                                               (setf ($prf mvs (+ (* nrm j) ii)) (* 1.0 nv)))))
                                          ((and (typep nv 'MX)
                                                (= ($nrow m) ($nrow nv))
                                                ($vector? nv))
                                           (let ((mvs ($ptr m))
                                                 (nrm ($nrow m))
                                                 (nvs ($ptr nv)))
                                             (dotimes (ii nrm)
                                               (setf ($prf mvs (+ (* nrm j) ii)) ($prf nvs ii)))))))
        ((and (numberp i) (eq j T)) (cond ((numberp nv) (let ((mvs ($ptr m))
                                                              (nrm ($nrow m))
                                                              (ncm ($ncol m)))
                                                          (dotimes (jj ncm)
                                                            (setf ($prf mvs (+ (* nrm jj) i))
                                                                  (* 1.0 nv)))))
                                          ((and (typep nv 'MX)
                                                (= ($ncol m) ($ncol nv))
                                                ($vector? nv))
                                           (let ((mvs ($ptr m))
                                                 (nrm ($nrow m))
                                                 (ncm ($ncol m))
                                                 (nvs ($ptr nv)))
                                             (dotimes (jj ncm)
                                               (setf ($prf mvs (+ (* nrm jj) i)) ($prf nvs jj)))))))
        ((and (numberp i) (numberp j)) (let* ((nr ($nrow m))
                                              (mvs ($ptr m)))
                                         (setf ($prf mvs (+ i (* nr j))) (* 1.0 nv))))))

(defsetf $ (m i j) (v) `($setv ,m ,v ,i ,j))

(defun $eye (&optional n)
  (cond ((null n) 1.0)
        ((= n 1) ($ones 1 1))
        (T (let ((zm ($zeros n n)))
             (dotimes (i n)
               (setf ($ zm i i) 1.0))
             zm))))

(defun $rows (m &optional indices)
  (if indices
      (let* ((nc ($ncol m))
             (nr ($count indices))
             (nm ($m 0 nr nc)))
        (loop :for i :from 0 :below nr :do (setf ($ nm i T) ($ m (elt indices i) T)))
        nm)
      m))

(defun $cols (m &optional indices)
  (if indices
      (let* ((nc ($count indices))
             (nr ($nrow m))
             (nm ($m 0 nr nc)))
        (loop :for j :from 0 :below nc :do (setf ($ nm T j) ($ m T (elt indices j))))
        nm)
      m))

(declaim (inline transpose-on))
(defun transpose-on (m nm)
  (let* ((nrm ($nrow m))
         (ncm ($ncol m))
         (mv ($fnv m))
         (nv ($fnv nm)))
    (dotimes (j ncm)
      (dotimes (i nrm)
        (setf ($ref nv (+ (* ncm i) j)) ($ref mv (+ (* nrm j) i)))))
    nm))

(defgeneric $transpose (m))

(defmethod $transpose ((n NUMBER)) n)
(defmethod $transpose ((m MX)) (transpose-on m ($m 0 ($ncol m) ($nrow m))))

(defun $sm (m i0 j0 nr nc &optional as-vector)
  (cond ((null as-vector) (let ((nm ($zeros nr nc)))
                            (dotimes (j nc)
                              (dotimes (i nr)
                                (setf ($ nm i j) ($ m (+ i0 i) (+ j0 j)))))
                            nm))
        (as-vector (let ((nm ($zeros 1 (* nr nc))))
                     (dotimes (j nc)
                       (dotimes (i nr)
                         (setf ($ nm 0 (+ (+ (* i nc) j))) ($ m (+ i0 i) (+ j0 j)))))
                     nm))))
