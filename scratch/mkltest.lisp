;; XXX not working, find how to load properly
(org.middleangle.load-blapack-libs::load-foreign-library "/usr/local/lib/libmkl.dylib")

(IN-PACKAGE :ORG.MIDDLEANGLE.CL-BLAPACK.BLAS-CFFI)
(cffi:defcfun ("SGEMM" %MKLSGEMM) :VOID (TRANSA :STRING) (TRANSB :STRING)
              (M FORTRAN-INT) (N FORTRAN-INT) (K FORTRAN-INT)
              (ALPHA FORTRAN-DOUBLE) (A CFFI-FNV-DOUBLE) (LDA FORTRAN-INT)
              (B CFFI-FNV-DOUBLE) (LDB FORTRAN-INT) (BETA FORTRAN-DOUBLE)
              (C CFFI-FNV-DOUBLE) (LDC FORTRAN-INT))

(CFFI:DEFCFUN ("SGEMM" %MKLSGEMM) :VOID (TRANSA :STRING) (TRANSB :STRING)
              (M FORTRAN-INT) (N FORTRAN-INT) (K FORTRAN-INT)
              (ALPHA FORTRAN-FLOAT) (A CFFI-FNV-FLOAT) (LDA FORTRAN-INT)
              (B CFFI-FNV-FLOAT) (LDB FORTRAN-INT) (BETA FORTRAN-FLOAT)
              (C CFFI-FNV-FLOAT) (LDC FORTRAN-INT))

(in-package :mtx)

(defun $mklgemm (a nra nca b nrb ncb &key (alpha 1.0D0) (beta 0.0D0) (c nil))
  (let* ((transa "N")
         (transb "N")
         (c (or c (make-fnv-float (* nra ncb))))
         (m (max 0 nra))
         (n (max 0 ncb))
         (k (max 0 (or nca nrb))))
    (org.middleangle.cl-blapack.blas-cffi::%mklsgemm transa transb m n k
                                                     alpha
                                                     a (max 1 m)
                                                     b (max 1 k)
                                                     beta
                                                     c (max 1 m))
    c))

($mklgemm ($fnv ($m '((1 2) (3 4)))) 2 2
          ($fnv ($m '((1 2) (3 4)))) 2 2)

($mm ($m '((1 2) (3 4))) ($m '((1 2) (3 4))))
