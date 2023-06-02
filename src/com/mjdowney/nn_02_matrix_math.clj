(ns com.mjdowney.nn-02-matrix-math
  "This second version is just like the first, but it introduces matrix math
  semantics.

  No speedup yet, because we're still using manual looping, multiplying and
  summing.")

(set! *warn-on-reflection* true)

;;; Matrix operations

(defn dot
  "Dot product of vectors"
  [v1 v2]
  (reduce + (map * v1 v2)))

(defn transpose
  "Transpose a matrix"
  [m]
  (apply mapv vector m))

(defn matmul
  "Multiply two matrices."
  [m1 m2]
  (vec
    (for [r1 m1]
      (vec
        (for [c2 (transpose m2)]
          (dot r1 c2))))))

(defn ewise
  "Build a function to apply `op` element-wise with two vectors or matrices,
  or against a single vector or matrix."
  [op]
  (fn operation
    ([v]
     (if (number? v)
       (op v)
       (mapv operation v)))
    ([v1 v2]
     (if (and (number? v1) (number? v2))
       (op v1 v2)
       (mapv operation v1 v2)))))

;;; Now the updated feedforward

; The sigmoid (σ) activation function and its derivative
(def sigmoid  (ewise (fn [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))))
(def sigmoid' (ewise (fn [z] (* (sigmoid z) (- 1 (sigmoid z))))))

(defn feedforward [weights biases inputs]
  (let [add (ewise +)]
    (loop [activations [inputs]
           zs []
           idx 0]
      (if (< idx (count weights))
        (let [w (nth weights idx)
              b (nth biases idx)
              z (add (matmul w (peek activations)) b)
              a (sigmoid z)]
          (recur (conj activations a) (conj zs z) (inc idx)))
        [zs activations]))))

; Same weights and biases from the tests in the previous namespace
(def test-weights
  [[[0.3130677, -0.85409574],
    [-2.55298982, 0.6536186],
    [0.8644362, -0.74216502]],
   [[2.26975462, -1.45436567, 0.04575852],
    [-0.18718385, 1.53277921, 1.46935877],
    [0.15494743, 0.37816252, -0.88778575],
    [-1.98079647, -0.34791215, 0.15634897]],
   [[1.23029068, 1.20237985, -0.38732682, -0.30230275]]])

(def test-biases
  [[[0.14404357], [1.45427351], [0.76103773]],
   [[0.12167502], [0.44386323], [0.33367433], [1.49407907]],
   [[-0.20515826]]])

^:rct/test
(comment
  ; Test that this matrix math version of feedforward is equivalent
  (let [[zs activations] (feedforward test-weights test-biases [[3] [4]])]
    (peek activations))
  ;=> [[0.7385495823882189]]
  )

;;; And the real challenge, the updated backpropagation

; Given some output delta (error)
; For each layer
;   Compute a change in weights based on the previous layer's activation
;   If not the first layer
;     Compute a change in the previous layer's biases based on this layer's weights
;     I.e. compute the previous layer's delta
;   Else if the first layer
;     Return weight and bias gradients
(defn backprop [weights biases inputs expected]
  (let [[zs as] (feedforward weights biases inputs)
        activation-for-layer (fn [l] (nth as (inc l)))
        multiply (ewise *)
        subtract (ewise -)]
    (loop [delta (multiply ; Given some output delta
                   (subtract (peek as) expected)
                   (sigmoid' (peek zs)))
           wg (list)
           bg (list delta)

           ; Iterate backwards over the layers
           layer (dec (count weights))]

      ; Compute a change in this layer's weights from this layer's delta and
      ; the previous layer's activation
      (let [w (matmul delta (transpose (activation-for-layer (dec layer))))
            wg (cons w wg)]
        ; If there is a preceding layer...
        (if-not (zero? layer)
          ; Compute a change in the previous layer's biases from this
          ; layer's weights / delta, and the previous layer's weighted
          ; activations
          (let [delta (multiply
                        (matmul (transpose (nth weights layer)) delta)
                        (sigmoid' (nth zs (dec layer))))
                bg (cons delta bg)]
            (recur delta wg bg (dec layer)))

          ; Otherwise if this is the first layer, return the gradients
          {:wg (vec wg)
           :bg (vec bg)})))))

^:rct/test
(comment
  (backprop test-weights test-biases [[3] [4]] [[5]])
  ;=>>
  {:wg [[[-0.14416451382871381 -0.19221935177161842]
         [0.0099253704585937 0.013233827278124935]
         [-0.226960031899466 -0.30261337586595466]]
        [[-0.021846114139705983 -0.006634547316343828 -0.14707552467992413]
         [-0.01435959524401721 -0.0043609318106062125 -0.09667371465695355]
         [0.006993662965455977 0.002123937811647067 0.04708373505242683]
         [0.003484428855991659 0.001058202294818807 0.023458368793990853]]
        [[-0.47480528631271807 -0.6525277264159817 -0.3763548135364771 -0.6604340885279466]]],
   :bg [[[-0.048054837942904605] [0.0033084568195312337] [-0.07565334396648866]]
        [[-0.24708450430178122] [-0.16241027810037612] [0.07909991388045874] [0.03940968041967343]]
        [[-0.8228609192012974]]]})

(defn backprop-batch [weights biases training-data]
  (let [add (ewise +)]
    (reduce
      ; Sum together all the weight and bias gradients in the batch
      (fn [[wg bg] backprop-results]
        (if-not wg
          ((juxt :wg :bg) backprop-results)
          [(add wg (:wg backprop-results))
           (add bg (:bg backprop-results))]))
      [nil nil]
      (pmap ; Do backprop on each training example in the batch
        (fn [{:keys [inputs outputs]}]
          (backprop weights biases inputs outputs))
        training-data))))

(defn train
  "Train the network `weights` and `biases` on the batch of `training-data`,
  returning updated weights and biases.

  The `training-data` is shaped [{:inputs [x] :outputs [y]} ...]."
  [weights biases training-data learning-rate]
  (let [[wg bg] (backprop-batch weights biases training-data)
        coef (/ learning-rate (count training-data))
        adjust (ewise
                 (fn [weight-or-bias gradient]
                   (- weight-or-bias (* gradient coef))))]
    [(adjust weights wg)
     (adjust biases bg)]))

^:rct/test
(comment
  (def training-data
    [{:inputs [[3] [4]] :outputs [[5]]}
     {:inputs [[4] [5]] :outputs [[6]]}
     {:inputs [[5] [6]] :outputs [[7]]}])

  (def trained
    (time
      (reduce
        (fn [[weights biases] td]
          (train weights biases td 0.001))
        [test-weights test-biases]
        (repeat 1000 training-data))))

  trained
  ;=>>
  [[[[0.42990755526687285 -0.7074824450899123]
     [-2.5556609160978114 0.6501218304459673]
     [0.9951770978037526 -0.5788488977828417]]
    [[2.2849577686829683 -1.4529221333741635 0.17692620896266537]
     [-0.17831988032714072 1.5336579052709052 1.5452033216566965]
     [0.15201595703752568 0.37785523163584306 -0.9144365699181696]
     [-1.9811838175395835 -0.3479784850885607 0.1513778801802436]]
    [[1.5501161477687693 1.637175464757474 -0.1758804998467201 0.11495297585370426]]]
   [[[0.17381700964321511] [1.4534478365437804] [0.7936129544134036]]
    [[0.2896804960563322] [0.5415332672690717] [0.29871588626034246] [1.4868532932245218]]
    [[0.31271354155253683]]]]

  (let [[_ as] (feedforward (first trained) (second trained) [[3] [4]])]
    (peek as))
  ;=> [[0.9439931067001217]]
  )
