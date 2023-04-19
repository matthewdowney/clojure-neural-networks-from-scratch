(ns com.mjdowney.faster-nn)

(set! *warn-on-reflection* true)

; The sigmoid (σ) activation function and its derivative
(defn sigmoid  [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))
(defn sigmoid' [z] (* (sigmoid z) (- 1 (sigmoid z))))

(defn feedforward
  "Activate each layer in the network in turn, returning a modified network."
  [network inputs]
  nil)

(defn activation
  "Return the activation of the final layer in the network."
  [network]
  nil)

(defn backprop
  "Feed the `inputs` through the `network`, compute the error versus the
  expected `outputs`, and step backwards through the network, estimating
  directional changes to the weights and biases of each layer."
  [network inputs outputs]
  nil)

(defn train
  "Train the `network` on the batch of `training-data`, returning a network
  with updated weights and biases."
  [network training-data learning-rate]
  nil)

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
  (def nn (network weights biases))

  ; Test feedforward result
  (activation (feedforward nn [3 4]))
  ;=> [0.7385495823882188]

  ; Test backprop result
  (backprop nn [3 4] [5])
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

  (def training-data
    [{:inputs [3 4] :outputs [5]}
     {:inputs [4 5] :outputs [6]}
     {:inputs [5 6] :outputs [7]}])

  ; The weight and bias updates are the same after 1000 batches of training as
  ; in the Python version
  (->> (repeat 1000 training-data)
       (reduce (fn [nn td] (train nn td 0.001)) nn)
       (mapv :neurons))
  ;=>>
  [[{:w [0.42990755526687285 -0.7074824450899123], :b 0.1738170096432151}
    {:w [-2.5556609160978114 0.6501218304459673], :b 1.4534478365437804}
    {:w [0.9951770978037526 -0.5788488977828417], :b 0.7936129544134036}]
   [{:w [2.2849577686829683 -1.4529221333741635 0.17692620896266537], :b 0.2896804960563322}
    {:w [-0.17831988032714072 1.5336579052709052 1.5452033216566965], :b 0.5415332672690717}
    {:w [0.15201595703752568 0.37785523163584306 -0.9144365699181696], :b 0.29871588626034246}
    {:w [-1.9811838175395835 -0.3479784850885607 0.15137788018024356], :b 1.4868532932245218}]
   [{:w [1.5501161477687693 1.637175464757474 -0.17588049984672005 0.1149529758537042], :b 0.31271354155253683}]]
  )
