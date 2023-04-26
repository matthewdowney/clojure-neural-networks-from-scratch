; apt-get install intel-mkl
(ns com.mjdowney.nn-matrix-math-fast
  (:require [clojure.java.io :as io]
            [uncomplicate.fluokitten.core :as fk]
            [uncomplicate.neanderthal.core :as ncore]
            [uncomplicate.neanderthal.native :as nnative]
            [uncomplicate.neanderthal.random :as nrand])
  (:import (java.util.zip GZIPInputStream)))

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

(defn mvec
  "Turn a Neanderthal matrix into a Clojure vector of vectors."
  [matrix]
  (vec
    (for [i (range (ncore/mrows matrix))]
      (vec
        (for [j (range (ncore/ncols matrix))]
          (ncore/entry matrix i j))))))

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
            b (nth biases idx)
            z (add (matmul w (peek activations)) b)
            a (sigmoid z)]
        (recur (conj activations a) (conj zs z) (inc idx)))
      [zs activations])))

; Same weights and biases from the tests in the previous namespace
(def test-weights
  [(nnative/dge 3 2
     [ 0.3130677 -0.85409574
      -2.55298982 0.6536186
      0.8644362 -0.74216502]
     {:layout :row})
   (nnative/dge 4 3
     [ 2.26975462 -1.45436567  0.04575852
      -0.18718385  1.53277921  1.46935877
      0.15494743  0.37816252 -0.88778575
      -1.98079647 -0.34791215  0.15634897]
     {:layout :row})
   (nnative/dge 1 4
     [ 1.23029068  1.20237985 -0.38732682 -0.30230275]
     {:layout :row})])

(def test-biases
  [(nnative/dge 3 1
     [0.14404357
      1.45427351
      0.76103773]
     {:layout :row})
   (nnative/dge 4 1
     [0.12167502
      0.44386323
      0.33367433
      1.49407907]
     {:layout :row})
   (nnative/dge 1 1
     [-0.20515826]
     {:layout :row})])

(def test-inputs
  (nnative/dge 2 1
    [3
     4]
    {:layout :row}))

^:rct/test
(comment
  ; Test that this matrix math version of feedforward is equivalent
  (let [[zs activations] (feedforward test-weights test-biases test-inputs)]
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
        multiply (element-wise *)
        subtract (element-wise -)]
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
  (-> (backprop test-weights test-biases
        (nnative/dge 2 1 [3 4] {:layout :col})
        (nnative/dge 1 1 [5] {:layout :col}))
      (update :wg #(mapv mvec %))
      (update :bg #(mapv mvec %)))
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
  (transduce
    (map ; Do backprop on each training example in the batch
      (fn [{:keys [inputs outputs]}]
        (backprop weights biases inputs outputs)))
    ; Sum together all the weight and bias gradients in the batch
    (completing
      (fn [[wg bg] backprop-results]
        (if-not wg
          ((juxt :wg :bg) backprop-results)
          [(mapv add wg (:wg backprop-results))
           (mapv add bg (:bg backprop-results))])))
    [nil nil]
    training-data))

(defrecord Network [weights biases])

(defn scale-and-add
  "Scale `m1` by the scalar `coef` and add it to `m2`.

  I.e. (+ (* m1 coef) + m2)"
  [coef m1 m2]
  (ncore/axpy coef m1 m2))

(defn train
  "Train the network `weights` and `biases` on the batch of `training-data`,
  returning updated weights and biases.

  The `training-data` is shaped [{:inputs [x] :outputs [y]} ...]."
  [{:keys [weights biases]} training-data learning-rate]
  (let [[wg bg] (backprop-batch weights biases training-data)
        coef (- (/ learning-rate (count training-data)))]
    (->Network
      (mapv #(scale-and-add coef %1 %2) wg weights)
      (mapv #(scale-and-add coef %1 %2) bg biases))))

^:rct/test
(comment
  (def training-data
    [{:inputs (nnative/dge 2 1 [3 4])
      :outputs (nnative/dge 1 1 [5])}
     {:inputs (nnative/dge 2 1 [4 5])
      :outputs (nnative/dge 1 1 [6])}
     {:inputs (nnative/dge 2 1 [5 6])
      :outputs (nnative/dge 1 1 [7])}])

  (def trained
    (time
      (reduce
        (fn [n td] (train n td 0.001))
        (->Network test-weights test-biases)
        (repeat 1000 training-data))))

  (update-vals trained #(mapv mvec %))
  ;=>
  {:weights [[[0.4299075552668728 -0.7074824450899123]
              [-2.5556609160978114 0.6501218304459673]
              [0.9951770978037526 -0.5788488977828417]]
             [[2.2849577686829683 -1.4529221333741635 0.1769262089626654]
              [-0.17831988032714072 1.5336579052709052 1.5452033216566965]
              [0.15201595703752568 0.37785523163584306 -0.9144365699181696]
              [-1.9811838175395835 -0.3479784850885607 0.15137788018024356]]
             [[1.5501161477687693 1.637175464757474 -0.17588049984672005 0.11495297585370423]]],
   :biases [[[0.1738170096432151] [1.4534478365437804] [0.7936129544134036]]
            [[0.2896804960563322] [0.5415332672690717] [0.29871588626034246] [1.4868532932245218]]
            [[0.31271354155253683]]]}

  (let [[_ as] (feedforward (:weights trained) (:biases trained)
                 (matrix [[3] [4]]))]
    (seq (peek as)))
  ;=> ((0.9439931067001217))
  )

;;; MNIST stuff

(defn evaluate [{:keys [weights biases]} test-data]
  (reduce + 0
    (map
      (fn [{:keys [inputs outputs]}]
        (let [[_ as] (feedforward weights biases inputs)
              output-vector (ncore/col (peek as) 0)]
          (if (= (ncore/iamax output-vector) outputs)
            1
            0)))
      test-data)))

(defn sgd [network training-data test-data eta]
  (let [start (System/currentTimeMillis)]
    (transduce
      (comp
        (partition-all 10)
        (map-indexed vector))
      (completing
        (fn [n [idx batch]]
          (let [n (train n batch eta)]
            (when (zero? (mod (inc idx) 1000))
              (println
                (format "Batch %s: accuracy %s / %s (t = %.3fs)"
                  idx
                  (evaluate n (take 100 (shuffle test-data)))
                  100
                  (/ (- (System/currentTimeMillis) start) 1000.0))))
            n)))
      network
      (shuffle training-data))))

(defn read-training-data-line [line]
  (let [[inputs outputs] (read-string line)
        inm (nnative/fge 784 1)
        om (nnative/fge 10 1)]
    (dotimes [i 784]
      (ncore/entry! inm i 0 (nth inputs i)))
    (dotimes [i 10]
      (ncore/entry! om i 0 (nth outputs i)))
    {:inputs inm
     :outputs om}))

(defn read-test-data-line [line]
  (let [[inputs outputs] (read-string line)
        inm (nnative/fge 784 1)]
    (dotimes [i 784]
      (ncore/entry! inm i 0 (nth inputs i)))
    {:inputs inm
     :outputs outputs}))

(comment
  (def mnist-training-data
    (let [path "resources/mnist/training_data.edn.gz"]
      (with-open [rdr (io/reader (GZIPInputStream. (io/input-stream path)))]
        (->> (line-seq rdr)
             (pmap read-training-data-line)
             (into [])))))

  (def mnist-test-data
    (let [path "resources/mnist/test_data.edn.gz"]
      (with-open [rdr (io/reader (GZIPInputStream. (io/input-stream path)))]
        (->> (line-seq rdr)
             (pmap read-test-data-line)
             (into [])))))

  ; Construct a network with 784 input neurons (for the 28 x 28 image pixels),
  ; a hidden layer of 30 neurons, and 10 output neurons (for the 10 digits).
  (def network
    (->Network
      [(nrand/rand-normal! 0 (/ 1 (Math/sqrt 724)) (nnative/fge 30 784))
       (nrand/rand-normal! 0 (/ 1 (Math/sqrt 30)) (nnative/fge 10 30))]
      [(nrand/rand-normal! 0 1 (nnative/fge 30 1))
       (nrand/rand-normal! 0 1 (nnative/fge 10 1))]))

  ; Initial accuracy is approximately random
  (evaluate network (take 100 (shuffle mnist-test-data))) ;=> 8

  ; Train the network for one epoch. This is approximately 100x faster than the
  ; version without using Neanderthal's native bindings (and slightly faster
  ; than the Python + NumPy version).
  (def trained
    (time
      (sgd network mnist-training-data mnist-test-data 3.0)))
  ; Batch 999: accuracy 74 / 100 (t = 1.322s)
  ; Batch 1999: accuracy 74 / 100 (t = 2.267s)
  ; Batch 2999: accuracy 80 / 100 (t = 3.203s)
  ; Batch 3999: accuracy 90 / 100 (t = 4.138s)
  ; Batch 4999: accuracy 87 / 100 (t = 5.244s)
  ; "Elapsed time: 3462.76129 msecs"

  ; After one epoch the accuracy on the test data is much higher...
  (evaluate trained mnist-test-data) ;=> 8996

  ; And you can just keep evaling this over an over again to train for
  ; additional epochs
  (dotimes [_ 10]
    (def trained
      (time
        (sgd trained mnist-training-data mnist-test-data 0.5))))

  (evaluate trained mnist-test-data) ;=> 9437
  )
