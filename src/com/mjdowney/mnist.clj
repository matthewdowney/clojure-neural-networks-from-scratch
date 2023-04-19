(ns com.mjdowney.mnist
  (:require [clojure.java.io :as io]
            [com.mjdowney.nn :as nn])
  (:import (java.util.zip GZIPInputStream)))

(set! *warn-on-reflection* true)

(defn load-data
  "Return a lazy sequence of data from the given `path`.

  The path should point to a gzipped file where each line is EDN data."
  [path]
  (letfn [(load-data* [lseq ^java.io.Closeable rdr]
            (lazy-seq
              (if-let [line (first lseq)]
                (cons (read-string line) (load-data* (rest lseq) rdr))
                (do (.close rdr) nil))))]
    (let [rdr (io/reader (GZIPInputStream. (io/input-stream path)))]
      (load-data* (line-seq rdr) rdr))))

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

(comment
  (defonce training-data (time (vec (load-data "resources/mnist/training_data.edn.gz"))))
  (def training-data-shuffled (shuffle training-data))

  (defonce test-data
    (time (vec (load-data "resources/mnist/test_data.edn.gz"))))

  (def network
    (nn/network
      [784 30 10]
      :af #'nn/sigmoid
      :af' #'nn/sigmoid'))

  ; Initial accuracy is 8 out of 100
  (evaluate network (take 100 (shuffle test-data))) ;=> 8

  (def trained
    (let [eta 3.0]
      (transduce
        (comp
          (map (fn [[i o]] {:inputs i :outputs o}))
          (partition-all 10)
          (take 1000)
          (map-indexed vector))
        (completing
          (fn [n [idx batch]]
            (let [n (nn/train n batch eta)]
              (when (zero? (mod idx 10))
              (println "Epoch" (str idx ":")
                (evaluate n (take 100 (shuffle test-data))) "/" 100))
              n)))
        network
        training-data-shuffled)))

  (time (evaluate network test-data))
  (time (evaluate trained test-data))
  )
