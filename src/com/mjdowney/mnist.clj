(ns com.mjdowney.mnist
  (:require [clojure.java.io :as io]
            [com.mjdowney.nn :as nn])
  (:import (java.util.zip GZIPInputStream)))

(set! *warn-on-reflection* true)

(defn load-data
  "Return a vector of data from the given `path`, which points to a gzipped
  file where each line is EDN data."
  [path]
  (with-open [rdr (io/reader (GZIPInputStream. (io/input-stream path)))]
    (->> (line-seq rdr)
         (pmap read-string)
         (into []))))

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

(defn evaluate [network test-data]
  (reduce + 0
    (pmap
      (fn [[inputs expected]]
        (let [a (nn/activation (nn/feedforward network inputs))]
          (if (= (argmax a) expected)
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
          (let [n (nn/train n batch eta)]
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
  (defonce training-data (load-data "resources/mnist/training_data.edn.gz"))
  (defonce test-data (load-data "resources/mnist/test_data.edn.gz"))

  ; Construct a network with 784 input neurons (for the 28 x 28 image pixels),
  ; a hidden layer of 30 neurons, and 10 output neurons (for the 10 digits).
  (def network (nn/network [784 30 10] :af #'nn/sigmoid :af' #'nn/sigmoid'))

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
