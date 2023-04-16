(ns com.mjdowney.nn-gist
  (:import (java.util Random)))

;;; I. Neurons
;;; ===
;;; A neuron is a function shaped [input] -> scalar.
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
;;; Notice that this structure can be represented as a matrix multiplication,
;;; so it goes even faster with modern SIMD instructions.
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

(defn network
  "Create a network with (deterministically) randomized weights and biases.

  For `sizes` [2 3 1], the network has two layers and takes 2 input values."
  [sizes]
  (let [r (Random. 0)
        rand (fn [] (.nextGaussian r))
        neuron (fn [inputs] {:w (vec (repeatedly inputs rand)) :b (rand)})]
    (mapv
      ; Each neuron has as many weights as the previous layer has neurons
      (fn [[inputs size]] (vec (repeatedly size (fn [] (neuron inputs)))))
      (partition 2 1 sizes))))

(defn feedforward-layer
  "Activate each neuron in the `layer` with the given `inputs` and activation
  function `afn`."
  [layer inputs afn]
  (mapv
    (fn [neuron]
      (let [z (reduce + (:b neuron) (map * (:w neuron) inputs))
            a (afn z)]
        (assoc neuron :z z :a a)))
    layer))

(defn feedforward
  "Activate each layer in the network in turn, returning a modified network.

  Takes a series of activation functions `afs` (one per layer).

  The return network is shaped [layer], each layer is shaped [neuron], and each
  neuron has an activation value :a and a weighted input value :z."
  [network inputs afs]
  (loop [idx 0
         inputs inputs
         network network]
    (if (< idx (count network))
      (let [layer (feedforward-layer (nth network idx) inputs (nth afs idx))]
        (recur
          (inc idx)
          (mapv :a layer)
          (assoc network idx layer)))
      network)))

; For backprop, I'll use :bg and :wg for bias gradient and weight gradient
; (typically these are written ∇b and ∇w, or nabla-b and nabla-w). I'm favoring
; short field names for REPL ergonomics.

; Each neuron changes its own weights, but only *contributes* to desired bias
; changes in the previous layer (its desired bias change is summed with the
; suggestions from other neurons at the same layer).
(defn backprop-neuron [neuron error prv-layer prv-layer-af']
  (assoc neuron
    ; From this neuron's perspective, the activations of the previous layer's
    ; neurons would have to change in this direction.
    :bg (mapv
          (fn [w z] (* w error (prv-layer-af' z)))
          (:w neuron) (map :z prv-layer))

    ; Change this neuron's weights in the direction of the error. E.g. if this
    ; neuron's activation needs to increase, increase to a greater degree the
    ; weights corresponding to *more active* neurons in the previous layer.
    :wg (mapv #(* (:a %) error) prv-layer)))

(defn backprop-layer-error [layer]
  ; If the neurons in the previous layer are [n0, n1, ..n], then
  ; this shape is [[n0', n1', ..n'], [n0'', n1'', ..n''], ...], with
  ; each top level vector representing the contribution from each neuron
  ; in *this* layer to deciding the error for the previous layer.
  (let [per-neuron-perspective (map :bg layer)]

    ; So define the error for each neuron in the previous layer as the sum of
    ; this layer's per-neuron assessments. I.e. sum each of the columns to get
    ; [(+ n0' n0'' ...) etc].
    (apply mapv + per-neuron-perspective)))

(defn network-output-error [network outputs afs']
  (let [activation-fn-derivative (peek afs')
        layer (peek network)]
    (mapv
      (fn [neuron output]
        (* (- (:a neuron) output)
           (activation-fn-derivative (:z neuron))))
      layer outputs)))

(defn backprop
  "Feed the `inputs` through the `network`, compute the error versus the
  expected `outputs`, and step backwards through the network, estimating
  directional changes to the weights and biases of each layer.

  Takes a series of activation functions `afs` and their derivatives `afs'`.

  Returns a map containing the weight and bias gradients (∇w and ∇b), shaped

     {:wg [[[weight-change]]]
      :bg [[bias-change]]}

  The nesting is a tad confusing. The top level vector corresponds to the
  layers, and then the next level to the neurons in each layer. There are
  multiple weights per neuron, hence the triple nesting."
  [network inputs outputs afs afs']
  (let [network (feedforward network inputs afs)
        input-layer (map (fn [a] {:a a}) inputs)]
    (loop [idx (dec (count network))
           network network
           error (list (network-output-error network outputs afs'))]
      (if (>= idx 0)
        (let [pl (nth network (dec idx) input-layer)
              ; activation function for the input layer is f(x) = x
              af' (nth afs' (dec idx) (constantly 1))
              layer (mapv #(backprop-neuron %1 %2 pl af')
                      (nth network idx)
                      (first error))]
          (recur
            (dec idx)
            (assoc network idx layer)
            (cons (backprop-layer-error layer) error)))
        {:bg (vec (rest error))
         :wg (mapv (partial mapv :wg) network)}))))

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
  (def nn
    (vec
      (for [[lws lbs] (map vector weights biases)]
        (vec
          (for [[nws [nb]] (map vector lws lbs)]
            {:w nws
             :b nb})))))

  ; Helper for building vectors of activation functions
  (defn fns [f] (vec (repeat 3 f)))

  ; Test feedforward result with ReLU
  (->> (feedforward nn [3 4] (fns relu)) peek (mapv :a))
  ;=> [0.711451179800545]

  ; Test backprop result with ReLU (ish, I cheated a bit on the derivative to
  ; get results with fewer zeros -- doesn't affect testing for consistency)
  (backprop nn [3 4] [5] (fns relu) (fns (constantly 1)))
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

  ; Test feedforward result with σ
  (->> (feedforward nn [3 4] (fns sigmoid)) peek (mapv :a))
  ;=> [0.7385495823882188]

  ; Test backprop result with σ
  (backprop nn [3 4] [5] (fns sigmoid) (fns sigmoid'))
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
  )
