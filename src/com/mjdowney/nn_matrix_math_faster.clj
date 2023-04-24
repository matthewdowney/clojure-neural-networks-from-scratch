; apt-get install intel-mkl
(ns com.mjdowney.nn-matrix-math-faster
  (:require [uncomplicate.fluokitten.core :as fk]
            [uncomplicate.neanderthal.core :as ncore]
            [uncomplicate.neanderthal.native :as nnative]))

(set! *warn-on-reflection* true)

;;; Matrix operations

(defn transpose
  "Transpose a matrix"
  [m]
  (ncore/trans m))

(defn matmul
  "Multiply two matrices."
  [m1 m2]
  (ncore/mm m1 m2))

(defn element-wise
  "Build a function to apply `op` element-wise with two vectors or matrices,
  or against a single vector or matrix."
  [op]
  (fk/fmap op))

; define add separately since there is a special helper for it
(defn add [m1 m2] (ncore/xpy m1 m2))

(defn repeat-vector
  "Repeat a single Neanderthal vector `n` times to create a matrix with `n`
  columns.

  So e.g. repeat a vector [3 4] 3 times to get :

      [3 3 3
       4 4 4]"
  [n v]
  (ncore/rk v (ncore/entry! (nnative/dv n) 1)))

;;; Now the updated feedforward

; The sigmoid (σ) activation function and its derivative
(def sigmoid  (element-wise (fn [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))))
(def sigmoid' (element-wise (fn [z] (* (sigmoid z) (- 1 (sigmoid z))))))

(defn feedforward [weights biases inputs]
  (loop [activations [inputs]
         zs []
         idx 0]
    (if (< idx (count weights))
      (let [w (nth weights idx)
            ; broadcast the biases into a matrix with as many columns as there
            ; are inputs, each column identical
            b (repeat-vector (ncore/ncols inputs) (ncore/col (nth biases idx) 0))
            z (add (matmul w (peek activations)) b)
            a (sigmoid z)]
        (recur (conj activations a) (conj zs z) (inc idx)))
      [zs activations])))

;; TODO: Improve the test vectors
; Same weights and biases from the tests in the previous namespace
(defn matrix [vecs] (ncore/trans (nnative/dge vecs)))
(defonce test-weights
  (mapv matrix
    [[[0.3130677, -0.85409574],
      [-2.55298982, 0.6536186],
      [0.8644362, -0.74216502]],
     [[2.26975462, -1.45436567, 0.04575852],
      [-0.18718385, 1.53277921, 1.46935877],
      [0.15494743, 0.37816252, -0.88778575],
      [-1.98079647, -0.34791215, 0.15634897]],
     [[1.23029068, 1.20237985, -0.38732682, -0.30230275]]]))

(defonce test-biases
  (mapv matrix
    [[[0.14404357], [1.45427351], [0.76103773]],
     [[0.12167502], [0.44386323], [0.33367433], [1.49407907]],
     [[-0.20515826]]]))

^:rct/test
(comment
  ; Test passing one input through the network
  (let [[zs as]
        (feedforward test-weights test-biases
          (nnative/dge 2 1
            [3
             4]
            {:layout :row}))]
    (seq (peek as)))
  ;=> ((0.7385495823882189))

  ; Test passing several inputs through the network
  (let [[zs as]
        (feedforward test-weights test-biases
          (nnative/dge 2 4
            [3 3 3 4
             4 4 4 5]
            {:layout :row}))]
    (seq (peek as)))
  ;=> ((0.7385495823882189) (0.7385495823882189) (0.7385495823882189) (0.7364765942503455))
  )
