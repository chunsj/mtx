(in-package :mtx)

;; https://medium.com/towards-data-science/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c

;; dot product
(let ((y ($m '(1 2 3)))
      (x ($m '(2 3 4))))
  ($* y ($transpose x)))

;; hadamard product
(let ((y ($m '(1 2 3)))
      (x ($m '(2 3 4))))
  ($x y x))

;; broadcasting...
(let ((a ($m '((1) (2))))
      (b ($m '((3 4) (5 6)))))
  ($x a b))

(let ((b ($m '((3 4) (5 6))))
      (c ($m '(1 2))))
  ($x b c))

(let ((a ($m '((1) (2))))
      (c ($m '(1 2))))
  ($+ a c))

(let ((a ($m '((2 3) (2 3))))
      (b ($m '((3 4) (5 6)))))
  ($x a b))

($transpose ($m '((1 2) (3 4))))

(let ((a ($m '((1 2))))
      (b ($m '((3 4) (5 6)))))
  ($* a b))
