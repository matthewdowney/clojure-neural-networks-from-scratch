(ns com.mjdowney.nn-matrix-math-fast
  (:require [uncomplicate.neanderthal.core :as ncore]
            [uncomplicate.neanderthal.native :as nnative]
            [uncomplicate.fluokitten.core :as fk]
            [uncomplicate.neanderthal.random :as nrand]))

(set! *warn-on-reflection* true)

;;; Matrix operations

(defn transpose "Transpose a matrix" [m] (ncore/trans m))

(defn matmul "Multiply two matrices." [m1 m2] (ncore/mm m1 m2))

(defn ewise
  "Build a function to apply `op` element-wise with two vectors or matrices,
  or against a single vector or matrix."
  [op]
  (fk/fmap op))

; define add separately since there is a special helper for it
(defn add [m1 m2] (ncore/xpy m1 m2))

;;; Now the updated feedforward

; The sigmoid (σ) activation function and its derivative
(def sigmoid  (ewise (fn [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))))
(def sigmoid' (ewise (fn [z] (* (sigmoid z) (- 1 (sigmoid z))))))

(defn feedforward [weights biases inputs]
  (loop [activations [inputs]
         zs []
         idx 0]
    (if (< idx (count weights))
      (let [w (nth weights idx)
            b (nth biases idx)
            z (add (matmul w (peek activations)) b)
            a (sigmoid z)]
        (recur (conj activations a) (conj zs z) (inc idx)))
      [zs activations])))

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
  ; Test that this matrix math version of feedforward is equivalent
  (let [[zs activations] (feedforward test-weights test-biases (matrix [[3] [4]]))]
    (seq (peek activations)))
  ;=> ((0.7385495823882189))
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
  (def wg (:wg *1))
  wg

  (-> (backprop test-weights test-biases
        (nnative/dge 2 1 [3 4] {:layout :col})
        (nnative/dge 1 1 [5] {:layout :col}))
      :wg
      count)

  (-> (backprop test-weights test-biases
        (nnative/dge 2 1 [3 4] {:layout :col})
        (nnative/dge 1 1 [5] {:layout :col}))
      (update :wg #(map seq %))
      (update :bg #(map seq %)))
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
  (reduce
    ; Sum together all the weight and bias gradients in the batch
    (fn [[wg bg] backprop-results]
      (if-not wg
        ((juxt :wg :bg) backprop-results)
        [(mapv add wg (:wg backprop-results))
         (mapv add bg (:bg backprop-results))]))
    [nil nil]
    (pmap ; Do backprop on each training example in the batch
      (fn [{:keys [inputs outputs]}]
        (backprop weights biases inputs outputs))
      training-data)))

(defrecord Network [weights biases])

(defn train
  "Train the network `weights` and `biases` on the batch of `training-data`,
  returning updated weights and biases.

  The `training-data` is shaped [{:inputs [x] :outputs [y]} ...]."
  [{:keys [weights biases]} training-data learning-rate]
  (let [[wg bg] (backprop-batch weights biases training-data)
        coef (- (/ learning-rate (count training-data)))]
    (->Network
      (mapv #(ncore/axpy coef %1 %2) wg weights)
      (mapv #(ncore/axpy coef %1 %2) bg biases))))

^:rct/test
(comment
  (def training-data
    [{:inputs (matrix [[3] [4]]) :outputs (matrix [[5]])}
     {:inputs (matrix [[4] [5]]) :outputs (matrix [[6]])}
     {:inputs (matrix [[5] [6]]) :outputs (matrix [[7]])}])

  (def trained
    (time
      (reduce
        (fn [n td] (train n td 0.001))
        (->Network test-weights test-biases)
        (repeat 1000 training-data))))

  (update-vals trained #(map seq %))
  ;=>
  {:weights '[((0.4299075552668728 -0.7074824450899123)
               (-2.5556609160978114 0.6501218304459673)
               (0.9951770978037526 -0.5788488977828417))
              ((2.2849577686829683 -1.4529221333741635 0.1769262089626654)
               (-0.17831988032714072 1.5336579052709052 1.5452033216566965)
               (0.15201595703752568 0.37785523163584306 -0.9144365699181696)
               (-1.9811838175395835 -0.3479784850885607 0.15137788018024356))
              ((1.5501161477687693 1.637175464757474 -0.17588049984672005 0.11495297585370423))],
   :biases '[((0.1738170096432151) (1.4534478365437804) (0.7936129544134036))
             ((0.2896804960563322) (0.5415332672690717) (0.29871588626034246) (1.4868532932245218))
             ((0.31271354155253683))]}

  (let [[_ as] (feedforward (:weights trained) (:biases trained)
                 (matrix [[3] [4]]))]
    (seq (peek as)))
  ;=> ((0.9439931067001217))
  )

;;; MNIST stuff

(defn evaluate [{:keys [weights biases]} test-data]
  (reduce + 0
    (pmap
      (fn [[inputs expected]]
        (let [[_ as] (feedforward weights biases inputs)
              output-vector (ncore/col (peek as) 0)]
          (if (= (ncore/iamax output-vector) expected)
            1
            0)))
      test-data)))

(defn sgd [network training-data test-data eta]
  (let [start (System/currentTimeMillis)]
    (transduce
      (comp
        (map (fn [[i o]] {:inputs i :outputs o}))
        (partition-all 10)
        (map-indexed vector))
      (completing
        (fn [n [idx batch]]
          (let [n (train n batch eta)]

            (when (zero? (mod idx 10))
              (println
                (format "Batch %s: accuracy %s / %s (t = %.3fs)"
                  idx
                  (evaluate n (take 100 (shuffle test-data)))
                  100
                  (/ (- (System/currentTimeMillis) start) 1000.0))))

            n)))
      network
      (shuffle training-data))))

(defn input-matrix
  "Convert a 784-element input vector to a 784x1 matrix."
  [input-vector]
  (nnative/dge 784 1 input-vector))

(defn output-matrix
  "Convert a 10-element output vector to a 10x1 matrix."
  [output-vector]
  (nnative/dge 10 1 output-vector))

(comment
  (def mnist-training-data
    (mapv
      (fn [[i o]]
        [(input-matrix i) (output-matrix o)])
      (com.mjdowney.mnist/load-data "resources/mnist/training_data.edn.gz")))

  (def mnist-test-data
    (mapv
      (fn [[i o]]
        [(input-matrix i) o])
      (com.mjdowney.mnist/load-data "resources/mnist/test_data.edn.gz")))

  ; Construct a network with 784 input neurons (for the 28 x 28 image pixels),
  ; a hidden layer of 30 neurons, and 10 output neurons (for the 10 digits).
  (def weights
    [(nrand/rand-normal! 0 1 (nnative/dge 30 784))
     (nrand/rand-normal! 0 1 (nnative/dge 10 30))])

  (def biases
    [(nrand/rand-normal! 0 1 (nnative/dge 30 1))
     (nrand/rand-normal! 0 1 (nnative/dge 10 1))])

  (def network (->Network weights biases))

  ; Initial accuracy is approximately random
  (evaluate network (take 100 (shuffle mnist-test-data))) ;=> 8

  (backprop
    (:weights network)
    (:biases network)
    (first (first mnist-training-data))
    (second (first mnist-training-data)))

  ; Train the network for one epoch -- this takes a long time because it goes
  ; through the full training data set!
  (def trained (sgd network mnist-training-data mnist-test-data 3.0))
  ; Batch 0: accuracy 4 / 100 (t = 0.156s)
  ; Batch 10: accuracy 16 / 100 (t = 0.914s)
  ; Batch 20: accuracy 13 / 100 (t = 1.665s)
  ; ...
  ; Batch 4970: accuracy 89 / 100 (t = 405.182s)
  ; Batch 4980: accuracy 90 / 100 (t = 405.926s)
  ; Batch 4990: accuracy 91 / 100 (t = 406.677s)

  ; After one epoch of training (consisting of many mini batches),
  ; the accuracy on the test data is > 90%. That's pretty cool!
  (evaluate trained mnist-test-data) ;=> 9109
  )
