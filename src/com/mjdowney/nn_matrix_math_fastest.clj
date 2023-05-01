; apt-get install intel-mkl
(ns com.mjdowney.nn-matrix-math-fastest
  (:require [uncomplicate.diamond.tensor :as tensor]
            [uncomplicate.diamond.dnn :as dnn]
            [clojure.java.io :as io]
            [uncomplicate.fluokitten.core :as fk]
            [uncomplicate.neanderthal.core :as ncore]
            [uncomplicate.neanderthal.native :as nnative]
            [uncomplicate.neanderthal.random :as nrand])
  (:import (java.util.zip GZIPInputStream)))

(set! *warn-on-reflection* true)

;;; MNIST stuff

#_(defn evaluate [{:keys [weights biases]} test-data]
  (reduce + 0
    (map
      (fn [[inputs expected]]
        (ccore/with-release [inputs (opencl/clge
                                      (ncore/mrows inputs)
                                      (ncore/ncols inputs)
                                      inputs)
                             result (feedforward weights biases inputs)]
          (let [[_ as] result
                output-vector (ncore/col (peek as) 0)]
            (if (= (ncore/iamax output-vector) expected)
              1
              0))))
      test-data)))

#_(defn sgd [network training-data test-data eta]
  (let [start (System/currentTimeMillis)
        idx (volatile! 0)]
    (reduce
      (fn [n [inputs outputs]]
        (ccore/with-release [inputs (opencl/clge
                                      (ncore/mrows inputs)
                                      (ncore/ncols inputs)
                                      inputs)
                             outputs (opencl/clge
                                       (ncore/mrows outputs)
                                       (ncore/ncols outputs)
                                       outputs)]
          (let [n (train n eta inputs outputs)]
            (when (zero? (mod (inc @idx) 1000))
              (println
                (format "Batch %s: accuracy %s / %s (t = %.3fs)"
                  @idx
                  (evaluate n (take 100 (shuffle test-data)))
                  100
                  (/ (- (System/currentTimeMillis) start) 1000.0))))
            (vswap! idx inc)
            n)))
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
  )

(set! *print-length* 128)
(def input-tz (tensor/tensor [10 1 28 28] :float :nchw))
(def output-tz (tensor/tensor [10 10] :float :nc))

(def net-bp
  (dnn/network input-tz
    [(dnn/fully-connected [30] :sigmoid)
     (dnn/fully-connected [10] :sigmoid)]))

(def net
  (dnn/init! (net-bp input-tz)))

(def costf
  (dnn/cost output-tz :mean-absolute))

(count (seq output-tz))

(dnn/train)

(let [[i o] (first mnist-training-data)]
  (ncore/transfer! i input-tz)
  (ncore/transfer! o output-tz)
  (dnn/infer net input-tz))
  ;(dnn/cost (dnn/infer net input-tz) output-tz :mean-absolute))

uncomplicate.diamond.internal.dnnl.directed

(comment

  ; Construct a network with 784 input neurons (for the 28 x 28 image pixels),
  ; a hidden layer of 30 neurons, and 10 output neurons (for the 10 digits).
  (clcore/with-default
    (opencl/with-default-engine
      (def network
        (->Network
          [(nrand/rand-normal! 0 (/ 1 (Math/sqrt 784)) (opencl/clge 30 784))
           (nrand/rand-normal! 0 (/ 1 (Math/sqrt 30)) (opencl/clge 10 30))]
          [(nrand/rand-normal! 0 1 (opencl/clge 30 1))
           (nrand/rand-normal! 0 1 (opencl/clge 10 1))]))
      ; Initial accuracy is approximately random
      (evaluate network (take 100 (shuffle mnist-test-data))))) ;=> 8

  ; Train the network for one epoch. This is approximately 100x faster than the
  ; version without using Neanderthal's native bindings (and slightly faster
  ; than the Python + NumPy version).
  (def trained
    (clcore/with-default
      (opencl/with-default-engine
        (time
          (sgd network mnist-training-data mnist-test-data 3.0)))))
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
        (sgd trained mnist-training-data mnist-test-data 0.5))))

  (evaluate trained mnist-test-data) ;=> 9604
  )
