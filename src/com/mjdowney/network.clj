(ns com.mjdowney.network
  (:import (java.util Random)))

(defprotocol INeuron
  (-nfeedforward [this inputs activationf])
  (-nbackprop [this desired-activation activationf']))

(defrecord Neuron [weights bias]
  INeuron
  (-nfeedforward [this inputs activationf]
    (let [weights-x-input (map * weights inputs)
          weighted-input (reduce + bias weights-x-input)]
      (assoc this
        :inputs inputs
        :feedforward/z weighted-input
        :feedforward/a (activationf weighted-input))))
  (-nbackprop [this desired-activation previous-layer]
    (let [a (:feedforward/a this)
          z (:feedforward/z this)
          delta (- a desired-activation)]
      (assoc this
        :backprop/delta delta
        ; The desired change to the previous layer's biases is given by the weights
        ; of this neuron, proportional to its activation error. This is just *this*
        ; neuron's vote. Its peer neurons also vote for how to define the previous
        ; layer's error; opinions from neurons experiencing the higher error are
        ; weighted more heavily.
        :backprop/bias-gradient
        (vec
          (for [weight weights]
            (* delta weight)))

        ; The desired change to this neuron's weights is in proportion to the
        ; activation of the previous layer, proportional to the error. Importantly,
        ; this changes the proportions between the weights, even though the error
        ; is constant, because the activation of the previous layer is (probably)
        ; not constant.
        :backprop/weight-gradient
        (vec
          (for [prev-neuron (:neurons previous-layer)]
            (* delta (:feedforward/a prev-neuron))))))))

(defprotocol ILayer
  (-lfeedforward [this inputs])
  (-lbackprop [this desired-activations previous-layer]))

(defrecord Layer [neurons activationf]
  ILayer
  (-lfeedforward [this inputs]
    (let [neurons (mapv (fn [n] (-nfeedforward n inputs activationf)) neurons)]
      (assoc this
        :neurons neurons
        :feedforward/a (mapv :feedforward/a neurons))))
  (-lbackprop [this desired-activations previous-layer]
    (let [neurons (mapv
                    (fn [n d] (-nbackprop n d previous-layer))
                    neurons #p desired-activations)
          ; If the neurons in the previous layer are [n0, n1, ..n], then
          ; this shape is [[n0', n1', ..n'], [n0'', n1'', ..n''], ...], with
          ; each top level vector representing the contribution for each neuron
          ; in *this* layer to deciding the error for the previous layer.
          next-delta (map :backprop/bias-gradient neurons)

          ; So sum each of the columns to get the error for each individual
          ; neuron in the previous layer: [(+ n0' n0'' ...) etc].
          next-delta #p (apply mapv + next-delta)]
      (assoc this
        :neurons neurons
        :backprop/prv-error next-delta
        :backprop/bias-gradient (mapv :backprop/delta neurons)
        :backprop/weight-gradient (mapv :backprop/weight-gradient neurons)))))

(defprotocol INetwork
  (feedforward [this inputs])
  (backprop [this inputs desired-activations]))

(defrecord Network [layers]
  INetwork
  (feedforward [this inputs]
    (loop [layers layers
           inputs inputs
           layers' []]
      (if-let [layer (first layers)]
        (let [layer' (-lfeedforward layer inputs)]
          (recur
            (rest layers)
            (:feedforward/a layer')
            (conj layers' layer')))
        (assoc this
          :layers layers'
          :feedforward/a inputs))))
  (backprop [this inputs error]
    (let [{:keys [layers]} (feedforward this inputs)]
      (loop [idx (dec (count layers))
             layers layers
             error error]
        (if (>= idx 0)
          (let [prv (if (zero? idx)
                      {:neurons (mapv (fn [a] {:feedforward/a a}) inputs)}
                      (nth layers (dec idx)))
                layer (-lbackprop (nth layers idx) error prv)]
            (recur
              (dec idx)
              (assoc layers idx layer)
              #p (:backprop/prv-error layer)))
          (assoc this
            :layers layers
            :backprop/bias-gradient (mapv :backprop/bias-gradient layers)
            :backprop/weight-gradient (mapv :backprop/weight-gradient layers)))))))

(defn sigmoid [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))
(defn sigmoid' [z] (* (sigmoid z) (- 1 (sigmoid z))))

(defn network [sizes]
  (let [r (Random. 0)
        rand (fn [] (.nextGaussian r))
        neuron (fn [inputs]
                 (let [weights (vec (repeatedly inputs rand))
                       bias (rand)]
                   (->Neuron weights bias)))]
    (->Network
      (mapv
        (fn [[n-inputs n-neurons]]
          (->Layer
            (vec (repeatedly n-neurons (fn [] (neuron n-inputs))))
            sigmoid))
        (partition 2 1 sizes)))))

(defn relu [z] (max 0 z))
(defn relu' [z] (if (pos? z) 1 0))

(defn network1 [weights biases]
  (->Network
    (vec
      (for [[lws lbs] (map vector weights biases)]
        (->Layer
          (vec
            (for [[nws [nb]] (map vector lws lbs)]
              (->Neuron nws nb)))
          relu)))))

(comment
  (def n (network [2 3 4 1]))

  (feedforward n [1 2])

  (backprop n [1 2] [3])
  )


^:rct/test
(comment
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

  (def nn (network1 weights biases))

  (-> nn (feedforward [3 4]) :feedforward/a) ;=> [0.711451179800545]

  (-> nn
      (backprop [3.0 4.0] [5.0])
      (select-keys [:backprop/bias-gradient :backprop/weight-gradient]))
  ;=>>
  {:backprop/bias-gradient
   [[-13.320990808641826 -0.0531463596480507 -9.090103139068606]
    [-5.276161644216386 -5.156464687149098 1.661069976942607 1.296440101855551]
    [-4.2885488201994555]],
   :backprop/weight-gradient
   [[[-39.96297242592548 -53.2839632345673]
     [-0.1594390789441521 -0.2125854385922028]
     [-27.27030941720582 -36.360412556274426]]
    [[-0.0 -0.0 -2.034942998951652]
     [-0.0 -0.0 -1.9887775284439588]
     [0.0 0.0 0.6406518503945805]
     [0.0 0.0 0.5000191212342855]]
    [[-0.5974954256335997 -4.333898954159115 -0.0 -6.666037594022326]]]}

  )

(comment
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

  (def nn (network1 weights biases))

  #p (-> nn
         (backprop [3.0 4.0] [-4.2885488201994555])
         (select-keys [:backprop/bias-gradient :backprop/weight-gradient]))
  )
