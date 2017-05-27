(in-package :mtx)

(declaim (optimize (speed 3) (safety 0) (debug 1)))

(defmacro println (&body body)
  `(print ,@body))

(defun $iota (n &key (start 0) (step 1))
  (alexandria:iota n :start start :step step))

(defun $shuffle (sequence)
  (map-into sequence #'car
            (sort (map 'vector (lambda (x)
                                 (cons x (random 1D0)))
                       sequence)
                  #'< :key #'cdr)))

(defun $partition (list cell-size)
  (loop :for cell :on list :by (lambda (list) (nthcdr cell-size list))
     :collecting (subseq cell 0 cell-size)))

(defun $linspace (from to length &key (endpoint T))
  (let ((step (/ (- to from) (if endpoint (1- length) length))))
    ($m (loop :for i :from 0 :below length
           :collect (coerce (+ from (* i step)) 'double-float)))))

(defmacro @> (initial-form &rest forms)
  (let ((output-form initial-form)
        (remaining-forms forms))
    (loop while remaining-forms do
         (let ((current-form (car remaining-forms)))
           (if (listp current-form)
	       (setf output-form (cons (car current-form)
				       (cons output-form (cdr current-form))))
	       (setf output-form (list current-form output-form))))
         (setf remaining-forms (cdr remaining-forms)))
    output-form))

(defmacro @>> (initial-form &rest forms)
  (let ((output-form initial-form)
        (remaining-forms forms))
    (loop while remaining-forms do
         (let ((current-form (car remaining-forms)))
	   (if (listp current-form)
	       (setf output-form (cons (car current-form)
				       (append (cdr current-form) (list output-form))))
	       (setf output-form (list current-form output-form))))
         (setf remaining-forms (cdr remaining-forms)))
    output-form))

(defun $str (&rest args)
  (if args
      (reduce (lambda (s0 s)
                (concatenate 'string
                             (cond ((stringp s0) s0)
                                   (T (write-to-string s0)))
                             (cond ((stringp s) s)
                                   (T (write-to-string s)))))
              args
              :initial-value "")
      ""))

(defun mkkw (s &optional (idx nil))
  (if idx
      (intern ($str (string-upcase s) idx) "KEYWORD")
      (intern (string-upcase s) "KEYWORD")))
