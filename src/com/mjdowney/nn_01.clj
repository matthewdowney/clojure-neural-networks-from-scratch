(ns com.mjdowney.nn-01
  "This first namespace builds up to a neural network that can learn digit
  recognition (MNIST), albeit slowly (~400 secs per epoch).

  Test vectors at the bottom of the ns."
  (:import (java.util Random)))

(set! *warn-on-reflection* true)

;;; I. Neurons
;;; ===
;;; A 'neuron' is a function shaped [input] -> scalar.
;;;
;;; To 'activate' the neuron, multiply the inputs by the neuron's weights (w),
;;; sum the results with its bias (b) to get the weighted input (z), and apply
;;; an activation function to get the neuron's output activation (a).

; Input_0 x Weight_0 \
; Input_1 x Weight_1  + Bias -> activationf(z) -> Output (Activation)
; Input_2 x Weight_2 /

; single neuron representing 'AND'
(def and-neuron {:w [1 1] :b -1.5})

; inputs are either 0 or 1, activation positive iff both inputs = 1
(let [activationf identity
      activate (fn [neuron inputs]
                 (activationf
                   (reduce + (:b neuron) (map * (:w neuron) inputs))))]
  (for [a [0 1] b [0 1]]
    {:inputs [a b] :a (activate and-neuron [a b])}))
;=> ({:inputs [0 0], :a -1.5}
;    {:inputs [0 1], :a -0.5}
;    {:inputs [1 0], :a -0.5}
;    {:inputs [1 1], :a 0.5})

;;; II. Networks
;;; ===
;;; A 'neural network' is a just a bunch of these neurons stacked together in
;;; layers, each layer shaped [neuron]. The first layer takes a vector of
;;; inputs, and subsequent layers take the previous layer's activation as
;;; inputs.
;;;
;;; This computation ('feedforward') is easy and fast, just sum and multiply.
;;; Notice that this can be represented as a matrix multiplication.
;;;
;;; The hard part is picking the right weights and biases.

;  I_0           I_1           I_2  Pass all inputs to all neurons
;    \ \         / \         / /    in the first layer.
;     \    \    /   \     /   /
;      \      \/     \ /     /
;       \     /  \  / \     /
;        \   /   /   \ \   /
;         \ / /        \\ /
;  L_0:  N_0_0        N_0_1         Each neuron takes three inputs and
;           \          /            (as always) yields one output. The
;            \        /             activations from the two neurons
;             \      /              in this layer become inputs for the
;              \    /               final layer.
;  L_1:        N_1_0

; The sigmoid (σ) activation function and its derivative
(defn sigmoid  [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))
(defn sigmoid' [z] (* (sigmoid z) (- 1 (sigmoid z))))

; The ReLU (rectified linear unit) activation function and its derivative
(defn relu  [z] (max z 0))
(defn relu' [z] (if (pos? z) 1 0))

; The hyperbolic tangent activation function and its derivative
(defn tanh  [z] (Math/tanh z))
(defn tanh' [z] (- 1 (Math/pow (Math/tanh z) 2)))

(defn layer
  "Create a map representing a layer of neurons.

  Also stores an activation function and its derivative."
  ([neurons]
   (layer neurons #'relu #'relu'))
  ([neurons activation-fn activation-fn']
   {:neurons neurons
    :af activation-fn
    :af' activation-fn'}))

(defn init-neurons
  "Create a vector of `out` neurons with randomized weights and biases, each
  taking `in` inputs.

  Pass an instance of `java.util.Random` as :random to get deterministic
  behavior."
  [& {:keys [in out random]}]
  (let [random (or random (Random.))
        rand (fn [] (.nextGaussian ^Random random))
        neuron (fn [inputs] {:w (vec (repeatedly inputs rand)) :b (rand)})]
    (vec (repeatedly out (fn [] (neuron in))))))

(defn network
  "Create a network with (deterministically) randomized weights and biases,
  shaped `[layer]`.

  For `sizes` [2 3 1], the network has two layers and takes 2 input values."
  [sizes & {:keys [af af'] :or {af #'relu af' #'relu'}}]
  (let [r (Random. 0)]
    (mapv
      (fn [[inputs outputs]]
        (layer (init-neurons :in inputs :out outputs :random r) af af'))
      (partition 2 1 sizes))))

(defn feedforward-layer
  "Activate each neuron in the `layer` with the given `inputs` and activation
  function."
  [{:keys [af neurons] :as layer} inputs]
  (assoc layer :neurons
    (mapv
      (fn [neuron]
        (let [z (reduce + (:b neuron) (map * (:w neuron) inputs))
              a (af z)]
          (assoc neuron :z z :a a)))
      neurons)))

(defn feedforward
  "Activate each layer in the network in turn, returning a modified network.

  The return network is shaped [layer], each layer is {:neurons [neuron]},
  and each neuron has an activation value :a and a weighted input value :z."
  [network inputs]
  (loop [idx 0
         inputs inputs
         network network]
    (if (< idx (count network))
      (let [layer (feedforward-layer (nth network idx) inputs)]
        (recur
          (inc idx)
          (mapv :a (:neurons layer))
          (assoc network idx layer)))
      network)))

(defn activation
  "Return the activation of the final layer in the network."
  [network]
  (mapv :a (:neurons (peek network))))

;;; III. Backpropagation
;;; ===
;;; To understand this algorithm, I'd recommend reading Michael Nielsen's
;;; tutorial through the end of chapter 1[1] and watching the 3Blue1Brown video
;;; on backprop[2].
;;  [1] http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
;;; [2] https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3

; I'll use :bg and :wg for bias gradient and weight gradient (typically these
; are written ∇b and ∇w, or nabla-b and nabla-w). I'm favoring short field names
; for REPL ergonomics.

; Each neuron changes its own weights, but only *contributes* to desired bias
; changes in the previous layer (its desired bias change is summed with the
; suggestions from other neurons at the same layer).
(defn backprop-neuron [neuron error {prv-neurons :neurons prv-af' :af'}]
  (assoc neuron
    ; From this neuron's perspective, the activations of the previous layer's
    ; neurons would have to change in this direction.
    :bg (mapv
          (fn [w z] (* w error (prv-af' z)))
          (:w neuron) (map :z prv-neurons))

    ; Change this neuron's weights in the direction of the error. E.g. if this
    ; neuron's activation needs to increase, increase to a greater degree the
    ; weights corresponding to *more active* neurons in the previous layer.
    :wg (mapv #(* (:a %) error) prv-neurons)))

(defn backprop-layer-error [layer]
  ; If the neurons in the previous layer are [n0, n1, ..n], then
  ; this shape is [[n0', n1', ..n'], [n0'', n1'', ..n''], ...], with
  ; each top level vector representing the contribution from each neuron
  ; in *this* layer to deciding the error for the previous layer.
  (let [per-neuron-perspective (map :bg (:neurons layer))]

    ; So define the error for each neuron in the previous layer as the sum of
    ; this layer's per-neuron assessments. I.e. sum each of the columns to get
    ; [(+ n0' n0'' ...) etc].
    (apply mapv + per-neuron-perspective)))

(defn network-output-error [network outputs]
  (let [layer (peek network)
        activation-fn-derivative (:af' layer)]
    (mapv
      (fn [neuron output]
        (* (- (:a neuron) output)
           (activation-fn-derivative (:z neuron))))
      (:neurons layer) outputs)))

(defn backprop
  "Feed the `inputs` through the `network`, compute the error versus the
  expected `outputs`, and step backwards through the network, estimating
  directional changes to the weights and biases of each layer.

  Returns a map containing the weight and bias gradients (∇w and ∇b), shaped

     {:wg [[[weight-change]]]
      :bg [[bias-change]]}

  The nesting is a tad confusing. The top level vector corresponds to the
  layers, and then the next level to the neurons in each layer. There are
  multiple weights per neuron, hence the triple nesting."
  [network inputs outputs]
  (let [network (feedforward network inputs)
        input-layer (layer
                      (mapv (fn [a] {:a a}) inputs)
                      ; activation function for the input layer is f(x) = x
                      identity (constantly 1))]
    (loop [idx (dec (count network))
           network network
           error (list (network-output-error network outputs))]
      (if (>= idx 0)
        (let [pl (nth network (dec idx) input-layer)
              layer (nth network idx)
              layer (update layer :neurons
                      (fn [ns]
                        (mapv #(backprop-neuron %1 %2 pl) ns (first error))))]
          (recur
            (dec idx)
            (assoc network idx layer)
            (cons (backprop-layer-error layer) error)))
        {:bg (vec (rest error))
         :wg (mapv (comp (partial mapv :wg) :neurons) network)}))))

; Helper for element-wise matrix operations
(defn matrix-op [f a b]
  (if (vector? a)
    (mapv #(matrix-op f %1 %2) a b)
    (f a b)))

(comment ; e.g.
  (matrix-op + [[1] [2 3]] [[3] [4 5]]) ;=> [[4] [6 8]]
  )

(defn backprop-batch
  "Do backprop with a whole batch of `training-data` at once, and return
  combined weight and bias gradients."
  [network training-data]
  (reduce
    ; Sum together all the weight and bias gradients in the batch
    (fn [[wg bg] backprop-results]
      (if-not wg
        ((juxt :wg :bg) backprop-results)
        [(matrix-op + wg (:wg backprop-results))
         (matrix-op + bg (:bg backprop-results))]))
    [nil nil]
    (pmap ; Do backprop on each training example in the batch
      (fn [{:keys [inputs outputs]}]
        (backprop network inputs outputs))
      training-data)))

(defn train
  "Train the `network` on the batch of `training-data`, returning a network
  with updated weights and biases.

  The `training-data` is shaped [{:inputs [x] :outputs [y]} ...]."
  [network training-data learning-rate]
  (let [[wg bg] (backprop-batch network training-data)
        coef (/ learning-rate (count training-data))]
    (vec
      (for [[{:keys [neurons] :as layer} wg bg] (map vector network wg bg)]
        (assoc layer :neurons
          (vec
            (for [[{:keys [w b] :as neuron} wg bg] (map vector neurons wg bg)]
              (let [weights (mapv (fn [w wg] (- w (* wg coef))) w wg)
                    bias (- b (* bg coef))]
                (assoc neuron :w weights :b bias)))))))))

;;; IV. MNIST Digit Recognition
;;; ===
;;; This code is actually enough to build a neural network that can recognize
;;; handwritten digits from the MNIST dataset. It's not very fast, but it works.

(defn load-data
  "Return a vector of data from the given `path`, which points to a gzipped
  file where each line is EDN data."
  [path]
  (with-open [rdr (clojure.java.io/reader
                    (java.util.zip.GZIPInputStream.
                      (clojure.java.io/input-stream path)))]
    (->> (line-seq rdr)
         (pmap read-string)
         (into []))))

; The network is going to have 10 output neurons, and we'll define its guess
; as the index of the neuron with the highest activation.
(defn argmax [xs]
  (loop [idx 1
         max-idx 0
         max-val (first xs)]
    (if (< idx (count xs))
      (let [val (nth xs idx)]
        (if (> val max-val)
          (recur (inc idx) idx val)
          (recur (inc idx) max-idx max-val)))
      max-idx)))

; Return the number of correct guesses the network makes on the test data.
(defn evaluate [network test-data]
  (reduce + 0
    (pmap
      (fn [[inputs expected]]
        (let [a (activation (feedforward network inputs))]
          (if (= (argmax a) expected)
            1
            0)))
      test-data)))

; Shuffle the training data and train the network using stochastic gradient
; descent with some learning rate `eta`.
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

(comment
  ; Load the training and test data
  (defonce training-data (load-data "resources/mnist/training_data.edn.gz"))
  (defonce test-data (load-data "resources/mnist/test_data.edn.gz"))

  ; Construct a network with 784 input neurons (for the 28 x 28 image pixels),
  ; a hidden layer of 30 neurons, and 10 output neurons (for the 10 digits).
  (def network (network [784 30 10] :af #'sigmoid :af' #'sigmoid'))

  ; Initial accuracy is approximately random
  (evaluate network (take 100 (shuffle test-data))) ;=> 8

  ; Train the network for one epoch -- this takes a long time because it goes
  ; through the full training data set!
  (def trained (sgd network training-data test-data 3.0))
  ; Batch 0: accuracy 4 / 100 (t = 0.156s)
  ; Batch 10: accuracy 16 / 100 (t = 0.914s)
  ; Batch 20: accuracy 13 / 100 (t = 1.665s)
  ; ...
  ; Batch 4970: accuracy 89 / 100 (t = 405.182s)
  ; Batch 4980: accuracy 90 / 100 (t = 405.926s)
  ; Batch 4990: accuracy 91 / 100 (t = 406.677s)

  ; After one epoch of training (consisting of many mini batches),
  ; the accuracy on the test data is > 90%. That's pretty cool!
  (evaluate trained test-data) ;=> 9109

  )

; Finally, I created some tests using Michael Nielsen's Python library[1].
; So I created a small NN with the same weights and biases that I'll put here,
; and trained it on the same data, and then compared the results to make sure
; that the code in this namespace is algorithmically correct.
;
; [1] https://github.com/mnielsen/neural-networks-and-deep-learning
;
^:rct/test
(comment
  ; Test that feedforward and backprop match Python library sample
  (def weights
    [[[0.3130677, -0.85409574],
      [-2.55298982, 0.6536186],
      [0.8644362, -0.74216502]],
     [[2.26975462, -1.45436567, 0.04575852],
      [-0.18718385, 1.53277921, 1.46935877],
      [0.15494743, 0.37816252, -0.88778575],
      [-1.98079647, -0.34791215, 0.15634897]],
     [[1.23029068, 1.20237985, -0.38732682, -0.30230275]]])

  (def biases
    [[[0.14404357], [1.45427351], [0.76103773]],
     [[0.12167502], [0.44386323], [0.33367433], [1.49407907]],
     [[-0.20515826]]])

  ; Network with the weights and biases taken from the Python library
  (defn create-nn [af af']
    (vec
      (for [[lws lbs] (map vector weights biases)]
        (layer
          (vec
            (for [[nws [nb]] (map vector lws lbs)]
              {:w nws
               :b nb}))
          af af'))))

  ; Test with relu-ish
  (def nn (create-nn #'relu (constantly 1)))

  ; Test feedforward result with ReLU
  (activation (feedforward nn [3 4]))
  ;=> [0.711451179800545]

  ; Test backprop result with ReLU (ish, I cheated a bit on the derivative to
  ; get results with fewer zeros -- doesn't affect testing for consistency)
  (backprop nn [3 4] [5])
  ;=>>
  {:bg [[-13.320990808641826 -0.0531463596480507 -9.090103139068606]
        [-5.276161644216386 -5.156464687149098 1.661069976942607 1.296440101855551]
        [-4.2885488201994555]],
   :wg [[[-39.96297242592548 -53.2839632345673]
         [-0.1594390789441521 -0.2125854385922028]
         [-27.27030941720582 -36.360412556274426]]
        [[-0.0 -0.0 -2.034942998951652]
         [-0.0 -0.0 -1.9887775284439588]
         [0.0 0.0 0.6406518503945805]
         [0.0 0.0 0.5000191212342855]]
        [[-0.5974954256335997 -4.333898954159115 -0.0 -6.666037594022326]]]}

  (def nn' (create-nn #'sigmoid #'sigmoid'))

  ; Test feedforward result with σ
  (activation (feedforward nn' [3 4]))
  ;=> [0.7385495823882188]

  ; Test backprop result with σ
  (backprop nn' [3 4] [5])
  ;=>
  {:bg [[-0.04805483794290462 0.003308456819531235 -0.07565334396648868]
        [-0.24708450430178128 -0.16241027810037614 0.07909991388045877 0.03940968041967344]
        [-0.8228609192012977]],
   :wg [[[-0.14416451382871387 -0.19221935177161847]
         [0.009925370458593704 0.01323382727812494]
         [-0.22696003189946604 -0.3026133758659547]]
        [[-0.021846114139705986 -0.006634547316343827 -0.14707552467992419]
         [-0.014359595244017212 -0.004360931810606212 -0.09667371465695357]
         [0.0069936629654559785 0.0021239378116470666 0.04708373505242684]
         [0.00348442885599166 0.0010582022948188068 0.023458368793990856]]
        [[-0.4748052863127182 -0.6525277264159819 -0.3763548135364772 -0.6604340885279468]]]}

  (def td
    [{:inputs [3 4] :outputs [5]}
     {:inputs [4 5] :outputs [6]}
     {:inputs [5 6] :outputs [7]}])

  ; The weight and bias updates are the same after 1000 batches of training as
  ; in the Python version
  (def trained
    (time
      (reduce
        (fn [nn td]
          (train nn td 0.001))
        nn'
        (repeat 1000 td))))

  (mapv :neurons trained)
  ;=>>
  [[{:w [0.42990755526687285 -0.7074824450899123], :b 0.1738170096432151}
    {:w [-2.5556609160978114 0.6501218304459673], :b 1.4534478365437804}
    {:w [0.9951770978037526 -0.5788488977828417], :b 0.7936129544134036}]
   [{:w [2.2849577686829683 -1.4529221333741635 0.17692620896266537], :b 0.2896804960563322}
    {:w [-0.17831988032714072 1.5336579052709052 1.5452033216566965], :b 0.5415332672690717}
    {:w [0.15201595703752568 0.37785523163584306 -0.9144365699181696], :b 0.29871588626034246}
    {:w [-1.9811838175395835 -0.3479784850885607 0.15137788018024356], :b 1.4868532932245218}]
   [{:w [1.5501161477687693 1.637175464757474 -0.17588049984672005 0.1149529758537042], :b 0.31271354155253683}]]

  (activation (feedforward trained [3 4]))
  ;=> [0.9439931067001217]
  )
