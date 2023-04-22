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

(defn cost-derivative [output-activations y] (np/subtract output-activations y))

; Given some output error
; For each layer
;   Compute a change in weights
;           (based on the previous layer's activation)
;   If not the first layer
;     Compute a change in the previous layer's biases
;           (based on this layer's weights)
;
;     Or, in other words
;
;     Compute the previous layer's error
;   Else if the first layer
;     Return weight and bias gradients
(defn backward [weights biases inputs expected]
  (let [[zs as] (forward weights biases inputs)
        activation-for-layer (fn [l] (nth as (inc l)))]
    (loop [error (np/multiply ; Given some output error
                   (np/subtract (peek as) expected)
                   (sigmoid' (peek zs)))
           wg (list)
           bg (list error)

           ; Iterate backwards over the layers
           layer (dec (count weights))]

      ; Compute a change in weights (based on the previous layer's activation)
      (let [w (np/dot error (np/transpose (activation-for-layer (dec layer))))
            wg (cons w wg)]
        ; If not the first layer
        (if-not (zero? layer)
          ; Compute a change in the previous layer's biases (based on this layer's
          ; weights and the previous layer's weighted activations)
          (let [error (np/multiply
                        (np/dot (np/transpose (nth weights layer)) error)
                        (sigmoid' (nth zs (dec layer))))
                bg (cons error bg)]
            (recur error wg bg (dec layer)))

          ; Else if the first layer
          ; Return weight and bias gradients
          {:wg (vec wg)
           :bg (vec bg)})))))


(defn round5 [x]
  (clojure.walk/postwalk
    (fn [n]
      (if (number? n)
        (/ (long (* n 1e5)) 1.0e5)
        n))
    x))

^:rct/test
(comment
  (-> (backward weights biases (np/array [[3] [4]]) (np/array [[5]]))
      pr-str
      read-string
      round5)
  ;=>>
  (round5
    {:bg [[[-0.04805483794290462] [0.003308456819531235] [-0.07565334396648868]]
          [[-0.24708450430178128] [-0.16241027810037614] [0.07909991388045877] [0.03940968041967344]]
          [[-0.8228609192012977]]],
     :wg [[[-0.14416451382871387 -0.19221935177161847]
           [0.009925370458593704 0.01323382727812494]
           [-0.22696003189946604 -0.3026133758659547]]
          [[-0.021846114139705986 -0.006634547316343827 -0.14707552467992419]
           [-0.014359595244017212 -0.004360931810606212 -0.09667371465695357]
           [0.0069936629654559785 0.0021239378116470666 0.04708373505242684]
           [0.00348442885599166 0.0010582022948188068 0.023458368793990856]]
          [[-0.4748052863127182 -0.6525277264159819 -0.3763548135364772 -0.6604340885279468]]]}))
