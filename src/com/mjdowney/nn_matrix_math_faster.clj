; apt-get install intel-mkl
(ns com.mjdowney.nn-matrix-math-faster
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

(defn repeat-vector
  "Repeat a single Neanderthal vector `n` times to create a matrix with `n`
  columns.

  So e.g. repeat a vector [3 4] 3 times to get :

      [3 3 3
       4 4 4]"
  [n v]
  (ncore/rk v (ncore/entry! (nnative/dv n) 1)))

(defn sum-rows
  "Sum the rows of a matrix and return a matrix of a single column.

  E.g. take
    [1 2
     3 4]
  and return
    [3
     7]"
  [m]
  (let [ones (ncore/entry! (nnative/dge (ncore/ncols m) 1) 1)]
    (ncore/mm m ones)))

(defn mvec
  "Turn a Neanderthal matrix into a Clojure vector of vectors."
  [matrix]
  (vec
    (for [i (range (ncore/mrows matrix))]
      (vec
        (for [j (range (ncore/ncols matrix))]
          (ncore/entry matrix i j))))))

;;; Now the updated feedforward
(let [idx+val (map-indexed vector ns)]
  ; the idx in [idx v] at which v is greatest
  (first (apply max-key second idx+val)))

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
            b (repeat-vector
                (ncore/ncols inputs)
                (ncore/col (nth biases idx) 0))
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

#_(def layers [{:weights w0 :biases b0} {:weights w1 :biases b1}])

; The sigmoid (σ) function to squash the output of the neurons onto [0, 1]
(defn sigmoid1 [n] (/ 1.0 (+ 1.0 (Math/exp (- n)))))

(defn feedforward1 [inputs {:keys [weights biases] :as _layer}]
  (for [[b ws :as _neuron] (map vector biases weights)]
    (let [weighted-input (reduce + (map * inputs ws))]
      (sigmoid (+ b weighted-input)))))

(defn argmax [numbers]
  (let [idx+val (map-indexed vector numbers)]
    ; the idx in [idx v] at which v is greatest
    (first (apply max-key second idx+val))))

(defn matrix->vecs [m]
  (vec
    (for [row (range (ncore/mrows m))]
      (vec
        (for [col (range (ncore/ncols m))]
          (ncore/entry m row col))))))

(comment
  (def layers
    (let [{:keys [weights biases]} trained]
      (for [[weights biases] (map vector weights biases)]
        {:weights (matrix->vecs weights)
         :biases (mapv first (matrix->vecs biases))})))

  (spit "wbs.txt" "")
  (doseq [i [0 1 2]
          k [:weights :biases]]
    (let [numbers (get (nth layers i) k)]
      (spit "wbs.txt"
        (prn-str
          (list 'def (symbol (str (first (name k)) i)) numbers))
        :append true)))

  (-> (repeat 784 1.0)
      (feedforward1 (first layers))
      (feedforward1 (second layers))
      #_(feedforward1 (last layers)))
  )

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
           bg (list (sum-rows delta))

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
                bg (cons (sum-rows delta) bg)]
            (recur delta wg bg (dec layer)))

          ; Otherwise if this is the first layer, return the gradients
          {:wg (vec wg)
           :bg (vec bg)})))))

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
  [{:keys [weights biases]} learning-rate inputs outputs]
  (let [{:keys [wg bg]} (backprop weights biases inputs outputs)
        coef (- (/ learning-rate (ncore/ncols inputs)))]
    (->Network
      (mapv #(scale-and-add coef %1 %2) wg weights)
      (mapv #(scale-and-add coef %1 %2) bg biases))))

^:rct/test
(comment
  (def training-inputs
    (nnative/dge 2 3
      [3 4 5
       4 5 6]
      {:layout :row}))

  (def training-outputs
    (nnative/dge 1 3 [5 6 7] {:layout :row}))

  (def trained
    (time
      (reduce
        (fn [n [ti to]] (train n 0.001 ti to))
        (->Network test-weights test-biases)
        (repeat 1000 [training-inputs training-outputs]))))

  (update-vals trained #(mapv mvec %))
  ;=>>
  {:weights [[[0.4299075552668728 -0.7074824450899123]
              [-2.5556609160978114 0.6501218304459673]
              [0.9951770978037526 -0.5788488977828417]]
             [[2.2849577686829683 -1.4529221333741635 0.17692620896266537]
              [-0.17831988032714072 1.5336579052709052 1.5452033216566965]
              [0.15201595703752568 0.37785523163584306 -0.9144365699181696]
              [-1.9811838175395835 -0.3479784850885607 0.1513778801802436]]
             [[1.5501161477687693 1.637175464757474 -0.17588049984672002 0.11495297585370425]]],
   :biases [[[0.1738170096432151] [1.4534478365437804] [0.7936129544134036]]
            [[0.2896804960563322] [0.5415332672690717] [0.29871588626034246] [1.4868532932245218]]
            [[0.31271354155253683]]]}

  (let [[_ as] (feedforward (:weights trained) (:biases trained)
                 (nnative/dge 2 1
                   [3
                    4]
                   {:layout :row}))]
    (seq (peek as)))
  ;=> ((0.9439931067001217))
  )

;;; MNIST stuff

(defn evaluate' [{:keys [weights biases]} test-data]
  (map
    (fn [[inputs expected]]
      (let [[_ as] (feedforward weights biases inputs)
            output-vector (ncore/col (peek as) 0)]
        (seq output-vector)))
    test-data))

(defn evaluate [{:keys [weights biases]} test-data]
  (reduce + 0
    (map
      (fn [[inputs expected]]
        (let [[_ as] (feedforward weights biases inputs)
              output-vector (ncore/col (peek as) 0)]
          (if (= (ncore/iamax output-vector) expected)
            1
            0)))
      test-data)))

(defn sgd [network training-data test-data eta]
  (let [start (System/currentTimeMillis)
        idx (volatile! 0)]
    (reduce
      (fn [n [inputs outputs]]
        (let [n (train n eta inputs outputs)]
          (when (zero? (mod (inc @idx) 1000))
            (println
              (format "Batch %s: accuracy %s / %s (t = %.3fs)"
                @idx
                (evaluate n (take 100 (shuffle test-data)))
                100
                (/ (- (System/currentTimeMillis) start) 1000.0))))
          (vswap! idx inc)
          n))
      network
      (shuffle training-data))))

(defn read-training-data-batch [lines]
  (let [batch-size (count lines)
        inm (nnative/dge 784 batch-size)
        om (nnative/dge 10 batch-size)]
    (dotimes [n batch-size]
      (let [[inputs outputs] (read-string (nth lines n))]
        (dotimes [m 784]
          (ncore/entry! inm m n (nth inputs m)))
        (dotimes [m 10]
          (ncore/entry! om m n (nth outputs m)))))
    [inm om]))

(defn read-test-data-line [line]
  (let [[inputs outputs] (read-string line)
        inm (nnative/dge 784 1)]
    (dotimes [i 784]
      (ncore/entry! inm i 0 (nth inputs i)))
    [inm outputs]))

(comment
  (def mnist-training-data
    (let [path "resources/mnist/training_data.edn.gz"]
      (with-open [rdr (io/reader (GZIPInputStream. (io/input-stream path)))]
        (into []
          (comp
            (partition-all 10)
            (map read-training-data-batch))
          (line-seq rdr)))))

  (def mnist-test-data
    (let [path "resources/mnist/test_data.edn.gz"]
      (with-open [rdr (io/reader (GZIPInputStream. (io/input-stream path)))]
        (->> (line-seq rdr)
             (pmap read-test-data-line)
             (into [])))))

  (spit "w.txt" (pr-str (list 'def 'inputs (vec (first (seq (ffirst mnist-test-data)))))))
  (-> mnist-test-data first second)
  (evaluate' network [(first mnist-test-data)])
  (partition-all 2)

  (evaluate' network (list [(nnative/dge 784 1 (repeat 1.0)) 0]))

  (->> inputs
       (partition-all 28)
       transpose
       (mapcat identity))

  ; Construct a network with 784 input neurons (for the 28 x 28 image pixels),
  ; a hidden layer of 30 neurons, and 10 output neurons (for the 10 digits).
  (def network
    (->Network
      [(nrand/rand-normal! 0 (/ 1 (Math/sqrt 784)) (nnative/dge 64 784))
       (nrand/rand-normal! 0 (/ 1 (Math/sqrt 64)) (nnative/dge 32 64))
       (nrand/rand-normal! 0 (/ 1 (Math/sqrt 32)) (nnative/dge 10 32))]
      [(nrand/rand-normal! 0 1 (nnative/dge 64 1))
       (nrand/rand-normal! 0 1 (nnative/dge 32 1))
       (nrand/rand-normal! 0 1 (nnative/dge 10 1))]))

  ; Initial accuracy is approximately random
  (evaluate network (take 100 (shuffle mnist-test-data))) ;=> 8

  ; Train the network for one epoch. This is approximately 100x faster than the
  ; version without using Neanderthal's native bindings (and slightly faster
  ; than the Python + NumPy version).
  (def trained
    (time
      (sgd network mnist-training-data mnist-test-data 3.0)))
  ; Batch 999: accuracy 88 / 100 (t = 0.326s)
  ; Batch 1999: accuracy 92 / 100 (t = 0.634s)
  ; Batch 2999: accuracy 90 / 100 (t = 0.946s)
  ; Batch 3999: accuracy 94 / 100 (t = 1.247s)
  ; Batch 4999: accuracy 95 / 100 (t = 1.584s)
  ; "Elapsed time: 1584.205381 msecs"

  ; After one epoch the accuracy on the test data is much higher...
  (evaluate trained mnist-test-data) ;=> 9413

  ; And you can just keep evaling this over an over again to train for
  ; additional epochs
  (dotimes [_ 10]
    (def trained
      (time
        (sgd trained mnist-training-data mnist-test-data 0.05))))

  (evaluate trained mnist-test-data) ;=> 9604

  (spit "w.txt" (prn-str (list 'def 'w0 (mapv vec (seq (first (:biases trained)))))))
  (mapv vec (seq (first (:biases trained))))
  (prn-str *1)
  )
