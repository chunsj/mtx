(in-package :mtx)

(declaim (optimize (speed 3) (safety 0) (debug 1)))

(defclass LAYER () ())

(defgeneric forward-propagate (layer &key)
  (:documentation "returns the result from the layer; generally :xs is input key"))
(defgeneric backward-propagate (layer &key)
  (:documentation "returns computed delta of the layer; generally with previous delta of :d"))
(defgeneric optimize-parameters (layer &key o)
  (:documentation "updates parameters using given optimizer"))
(defgeneric regularization (layer)
  (:documentation "returns computed regularization value to avoid overfitting"))
(defgeneric parameters (layer)
  (:documentation "returns the list of parameters of the layer"))

;; dummy, base implementation
(defmethod forward-propagate ((l LAYER) &key))
(defmethod backward-propagate ((l LAYER) &key))
(defmethod optimize-parameters ((l LAYER) &key o) (declare (ignore o)))
(defmethod regularization ((l LAYER)) 0.0)
(defmethod parameters ((l LAYER)) nil)

(defclass OPTIMIZER () ())

(defgeneric initialize-parameters (optimizer params)
  (:documentation "prepares internal data for given parameters"))
(defgeneric update (optimizer params gradients)
  (:documentation "updates parameters with gradient information; updates should be in-place ones"))

(defmethod initialize-parameters ((o OPTIMIZER) params) (declare (ignore params)))

(defclass NEURALNETWORK () ())

(defgeneric predict (network &key)
  (:documentation "returns result of forward propagations of layers"))
(defgeneric loss (network &key)
  (:documentation "returns computed error"))
(defgeneric gradient (network &key)
  (:documentation "computes gradients of parameters using backward propagation"))
(defgeneric train (network &key)
  (:documentation "updates parameters to have improved values for better fit; single step"))
