(ns com.mjdowney.scratch
  (:require [libpython-clj2.python :as py]
            [libpython-clj2.require :refer [require-python]]))

(set! *warn-on-reflection* true)

(do
  (py/initialize!)
  (require-python '[numpy :as np]))

(def weights
  (mapv np/array
    [[[0.3130677, -0.85409574],
      [-2.55298982, 0.6536186],
      [0.8644362, -0.74216502]],
     [[2.26975462, -1.45436567, 0.04575852],
      [-0.18718385, 1.53277921, 1.46935877],
      [0.15494743, 0.37816252, -0.88778575],
      [-1.98079647, -0.34791215, 0.15634897]],
     [[1.23029068, 1.20237985, -0.38732682, -0.30230275]]]))

(def biases
  (mapv np/array
    [[[0.14404357], [1.45427351], [0.76103773]],
     [[0.12167502], [0.44386323], [0.33367433], [1.49407907]],
     [[-0.20515826]]]))

(defn sigmoid [z]
  (np/divide 1 (np/add 1 (np/exp (np/negative z)))))

(defn sigmoid' [z]
  (np/multiply (sigmoid z) (np/subtract 1 (sigmoid z))))

(defn forward [weights biases inputs]
  (loop [activations [inputs]
         zs []
         idx 0]
    (if (< idx (count weights))
      (let [w (nth weights idx)
            b (nth biases idx)
            z (np/add (np/dot w (peek activations)) b)
            a (sigmoid z)]
        (recur
          (conj activations a)
          (conj zs z)
          (inc idx)))
      [zs activations])))

(defn cost-derivative [output-activations y]
  (np/subtract output-activations y))

(defn ? [v idx] (if (>= idx 0) (nth v idx) (? v (+ (count v) idx))))

(defn backward [weights biases x y]
  (let [[zs as] (forward weights biases x)
        n-layers (inc (count weights))]
    (loop [delta (np/multiply
                   (cost-derivative (peek as) y)
                   (sigmoid' (peek zs)))
           nabla-b (list delta)
           nabla-w (list (np/dot delta (np/transpose (? as -2))))
           idx 2]
      (if (< idx n-layers)
        (let [z (? zs (- idx))
              sp (sigmoid' z)
              delta (np/multiply
                      (np/dot
                        (np/transpose (? weights (+ (- idx) 1)))
                        delta)
                      sp)
              w (np/dot delta (np/transpose (? as (- (- idx) 1))))]
          (recur delta (cons delta nabla-b) (cons w nabla-w) (inc idx)))
        [nabla-b nabla-w]))))

(comment
  (backward weights biases (np/array [[3] [4]]) (np/array [[5]]))
  )
