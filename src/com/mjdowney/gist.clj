(ns com.mjdowney.gist
  (:import (java.util Random)))

;; A single "neuron" is shaped
;;
;; Input_0  _
;;           \ (Weight_0)
;;            \
;;             \ (bias)
;;             | ———> Neuron ———> Activation
;;            /
;;           /
;;          / (Weight_N)
;; Input_N -
;;
;; So the N input numbers are multiplied by the N weights, and then
;; summed with the scalar bias to get the output (activation).

(defrecord Neuron [weights bias inputs activation])
(defn n [weights bias] (->Neuron weights bias nil nil))

(defn activate [neuron inputs]
  ; The weighted input is `weights x inputs + bias` (where `x` is `dot`)
  (let [weights-x-input (map * (:weights neuron) inputs)
        weighted-input (reduce + (:bias neuron) weights-x-input)]
    (assoc neuron
      :inputs inputs
      :activation weighted-input)))

; A single neuron can implement simple functions, like AND.
; Remember, it's going to take two inputs, so it needs two weights.
(def and-neuron (n [1 1] -1.5))

; Because the weights and bias are set just so, the activation is only
; positive if both inputs are 1 (assuming inputs are 0 or 1)
(activate and-neuron [1 1]) ;=> {:activation 0.5}
(activate and-neuron [1 0]) ;=> {:activation -0.5}

(for [x [0 1] y [0 1]]
  [x y (-> (activate and-neuron [x y]) :activation pos?)])
;=> ([0 0 false] [0 1 false] [1 0 false] [1 1 true])

;; A "network" of neurons is composed of layers of neurons.
;;
;;  I_0           I_1           I_2  Pass all inputs to all neurons
;;    \ \         / \         / /    in the first layer.
;;     \    \    /   \     /   /
;;      \      \/     \ /     /
;;       \     /  \  / \     /
;;        \   /   /   \ \   /
;;         \ / /        \\ /
;;  L_0:  N_0_0        N_0_1         Each neuron takes three inputs and
;;           \          /            (as always) yields one output. The
;;            \        /             activations from the two neurons
;;             \      /              in this layer become inputs for the
;;              \    /               final layer.
;;  L_1:        N_1_0
;;
;; Computation is simple: just multiply and sum.
;;
;; The trick is to pick the weights and biases so that the final layer
;; activates to a useful output.

;;; Now let's try a "network" of neurons to take [x, y] and output √x^2 + y^2
;;; Each "layer" of the network is composed of some series of neurons.
;;; The first layer takes the inputs, and then the set of activations from
;;; the first layer is the input for the next, and so on. The last layer's
;;; activations are the output of the network.

; Each layer in the network is shaped [neuron]
(defrecord Network [layers])

; Initialize a network with random weights and biases
(def network
  (let [r (Random. 0)
        neuron (fn [inputs]
                 (->Neuron
                   (vec (take inputs (repeatedly #(.nextDouble r))))
                   (.nextDouble r)
                   nil nil))]
    (->Network
      ; First layer of three takes two inputs
      [[(neuron 2) (neuron 2) (neuron 2)]
       ; Next layer of four takes three inputs
       [(neuron 3) (neuron 3) (neuron 3) (neuron 3)]
       ; Final layer takes 4 inputs and outputs a single number
       [(neuron 4)]])))

; Use a slightly different activation fn with activation >= 0
(defn activate [neuron inputs]
  (let [weights-x-input (map * (:weights neuron) inputs)
        weighted-input (reduce + (:bias neuron) weights-x-input)]
    (assoc neuron
      :inputs inputs
      :activation (max 0 weighted-input)))) ; <- only difference

; Feeding a signal through the network is just multiplying each layer
(defn compute-network [{:keys [layers] :as network} inputs]
  (let [inputs (volatile! inputs)
        layers (mapv
                 (fn [layer]
                   ; Update the neurons in the layer for the inputs
                   (let [layer' (mapv #(activate % @inputs) layer)]
                     ; The activations become the next inputs
                     (vreset! inputs (mapv :activation layer'))
                     layer'))
                 layers)]
    (assoc network :layers layers)))

; Inputting [3 4], we expect √3^2 + 4^2 = 5
(def network' (compute-network network [3 4]))

; ... but because the weights and biases are random, we do not get that
(-> network' :layers peek peek)
;=> {:activation 19.442854771638164 ...}

; Time to improve the weights and biases until the answer is correct.

; To back-propagate the error through the network, start with the delta
; between the result and the desired result
(def layers (:layers network'))

; a 'get' which allows negative indexes for vector positions
(defn ? [v i] (if (neg? i) (nth v (+ i (count v))) (nth v i)))

; one output, so deltas is a vector ∈ R1
(def deltas [(mapv - (map :activation (? layers -1)) [5.0])])

; TODO: once this works, try a version going neuron by neuron
; we have to introduce matrix multiplication here
(defn t [m] (apply mapv vector m))
(defn dot [ma mb]
  (vec
    (for [row ma]
      (vec
        (for [col (t mb)]
          (reduce + (map * row col)))))))

(dot deltas [(mapv :activation (? layers -2))])

; Store the gradients in list, to append to the front as we work backwards
(def backprop-results
  (let [activations (fn [idx] [(mapv :activation (? layers idx))])
        weights (fn [idx] (mapv :weights (? layers idx)))
        bias-gradient (list deltas)
        weight-gradient (list (dot deltas (t (activations -2))))]
    (reduce
      (fn [[bg wg d] idx]
        (let [delta (dot (t (weights (+ (- idx) 1))) d)
              weight (dot delta (activations (+ (- idx) 1)))]
          [(cons delta bg) (cons weight wg) delta]))
      [bias-gradient weight-gradient deltas]
      (range 2 (inc (count layers))))))

; The eta symbol "η" represents the learning rate
(def η 0.001)

; Update the network for the bias and weight gradients
(def network''
  (assoc network' :layers
    (vec
      (for [[neurons ∇b ∇w] (map vector (:layers network')
                              (first backprop-results)
                              (second backprop-results))]
        (vec
          (for [[[b] [w] n] (map vector ∇b ∇w neurons)]
            (let [change (* w η)]
              (assoc n
                :weights (mapv #(- % change) (:weights n))
                :bias (+ (:bias n) (* b η))))))))))

; Alright ... after the updates, the result is closer
(-> (compute-network network'' [3 4]) :layers peek peek :activation)
;=> 8.83472602134939

; compared to
(-> (compute-network network' [3 4]) :layers peek peek :activation)
;=> 19.442854771638164

;;;

(defn backprop [network inputs expected]
  (let [{:keys [layers] :as network} (compute-network network inputs)
        deltas [(mapv - (map :activation (? layers -1)) expected)]
        activations (fn [idx] (mapv (comp vector :activation) (? layers idx)))
        weights (fn [idx] (mapv :weights (? layers idx)))
        bias-gradient (list deltas)
        weight-gradient (list (dot deltas (t (activations -2))))

        [∇b ∇w]
        (reduce
          (fn [[bg wg d] idx]
            (let [delta (dot (t (weights (- idx))) d)
                  avs (if (< idx (dec (count layers)))
                        (activations (- (+ idx 2)))
                        [inputs])
                  weight (dot delta (t avs))]
              [(cons delta bg) (cons weight wg) delta]))
          [bias-gradient weight-gradient deltas]
          ; plus one bc of the input layers
          (range 1 (count layers)))]
    {:∇b ∇b :∇w ∇w :network network :diff (ffirst deltas)}))

(defn train [network samples η]
  (letfn [(ele-plus [a b]
            (if-not a
              b
              (if (number? a)
                (+ a b)
                (mapv ele-plus a b))))]
    (let [[network ∇b ∇w dsq]
          (reduce
            (fn [[network ∇b ∇w dsq] {:keys [inputs expected]}]
              (let [ret (backprop network inputs expected)
                    ∇b (ele-plus ∇b (:∇b ret))
                    ∇w (ele-plus ∇w (:∇w ret))]
                [(:network ret) ∇b ∇w (+ dsq (* (:diff ret) (:diff ret)))]))
            [network nil nil 0.0]
            samples)

          effective-η (/ η (count samples))]
      (assoc network
        :cost (/ dsq (* (count samples) 2))
        :layers
        (vec
          (for [[neurons ∇b ∇w] (map vector (:layers network) ∇b ∇w)]
            (vec
              (for [[[b] [w] n] (map vector ∇b ∇w neurons)]
                (let [dweight (* w effective-η)]
                  (assoc n
                    :weights (mapv #(- % dweight) (:weights n))
                    :bias (+ (:bias n) (* b effective-η))))))))))))

(def training-batches
  (let [r (Random. 0)]
    (repeatedly
      (fn []
        (vec
          (for [_ (range 10)
                :let [a (+ 0.1 (* (.nextDouble r) 50.0))
                      b (+ 0.1 (* (.nextDouble r) 50.0))]]
            {:inputs [a b]
             :expected [(Math/sqrt (+ (Math/pow a 2) (Math/pow b 2)))]}))))))

(def trained
  (loop [n network
         idx+data (map-indexed vector training-batches)]
    (let [[idx data] (first idx+data)]
      (if (< idx 100)
        (let [trained (train n data 0.0001)]
          (println "epoch" idx "cost=" (:cost trained))
          (recur trained (rest idx+data)))
        n))))

^:rct/test
(comment
  (compute-network trained [49 49.05])
  (let [[x y] [49 49.05]]
    (Math/sqrt (+ (* x x) (* y y))))

  (let [[x y :as input] [40 50]]
    [(-> (compute-network trained input) :layers peek peek :activation)
     (Math/sqrt (+ (* x x) (* y y)))])
  ;=> [64.61321601023403 64.03124237432849]

  (let [[x y :as input] [41.33 35.035]]
    [(-> (compute-network trained input) :layers peek peek :activation)
     (Math/sqrt (+ (* x x) (* y y)))])
  ;=> [55.000536143547976 54.181363262657015]
  )


^:rct/test
(comment
  ; Test values to make sure the feed forward and back prop match ready-made
  ; Python library
  (def weights
    [[[0.3130677 , -0.85409574],
      [-2.55298982, 0.6536186],
      [0.8644362 , -0.74216502]],
     [[2.26975462, -1.45436567, 0.04575852],
      [-0.18718385, 1.53277921, 1.46935877],
      [0.15494743, 0.37816252, -0.88778575],
      [-1.98079647, -0.34791215, 0.15634897]],
     [[1.23029068, 1.20237985, -0.38732682, -0.30230275]]])

  (def biases
    [[[0.14404357], [1.45427351], [0.76103773]],
     [[0.12167502], [0.44386323], [0.33367433], [1.49407907]],
     [[-0.20515826]]])

  (def nn
    (->Network
      (vec
        (for [[lws lbs] (map vector weights biases)]
          (vec
            (for [[nws [nb]] (map vector lws lbs)]
              (->Neuron nws nb nil nil)))))))

  (-> (compute-network nn [3 4]) :layers peek peek :activation)
  ;=> 0.711451179800545

  (def expected-gradient-vectors
    {:∇b [[[-13.320990808641826] [-0.0531463596480507] [-9.090103139068606]]
          [[-5.276161644216386] [-5.156464687149098] [1.661069976942607] [1.296440101855551]]
          [[-4.2885488201994555]]],
     :∇w [[[-39.96297242592548] [-0.1594390789441521] [-27.27030941720582]]
          [[-0.0 -0.0 -2.034942998951652]
           [-0.0 -0.0 -1.9887775284439588]
           [0.0 0.0 0.6406518503945805]
           [0.0 0.0 0.5000191212342855]]
          [[-0.5974954256335997 -4.333898954159115 -0.0 -6.666037594022326]]]})

  ; Expected gradient vectors for one round of backprop
  (dissoc (backprop nn [3 4] [5]) :network :diff)
  ;=>> expected-gradient-vectors
  )
