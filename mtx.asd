(asdf:defsystem mtx
  :name "mtx"
  :author "Sungjin Chun <chunsj@gmail.com>"
  :version "0.1"
  :maintainer "Sungjin Chun <chunsj@gmail.com>"
  :license "GPL3"
  :description "simple follow up of deep learning from scratch book"
  :long-description "trying to create a helper code in common lisp using cl-blapack for dlfs book"
  :depends-on ("org.middleangle.cl-blapack"
               "org.middleangle.foreign-numeric-vector"
               "alexandria")
  :components ((:file "package")
               (:file "matrix")
               (:file "basic")
               (:file "funcs")
               (:file "utils")
               (:file "mnist")
               (:file "nnbs")
               (:file "layers")
               (:file "optimizers")
               (:file "nn")))
