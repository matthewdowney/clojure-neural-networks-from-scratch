; Exclude basic arithmetic when including Clojure core:
(ns com.mjdowney.micrograd
  (:require [clojure.java.io :as io]
            [nextjournal.clerk :as clerk]
            [nextjournal.clerk.viewer :as v]
            [com.phronemophobic.clj-graphviz :as gv])
  (:refer-clojure :exclude [+ - * /])
  (:import (javax.imageio ImageIO)))

{:nextjournal.clerk/visibility {:code :show :result :hide}}

; Create a custom `Value` type that implements the protocol
(defrecord Value [v op children])

(defn -op [this other f]
  (cond
    (map? this)
    (map->Value
      {:v (f (:v this) (:v other))
       :op (symbol (name (symbol f)))
       :children #{this other}})

    (instance? java.lang.Number this) (f this other)
    :else (f other this)))

(defn + [this other] (-op this other #'clojure.core/+))
(defn - [this other] (-op this other #'clojure.core/-))
(defn * [this other] (-op this other #'clojure.core/*))
(defn / [this other] (-op this other #'clojure.core//))

(defn value [v] (->Value v nil #{}))

{:nextjournal.clerk/visibility {:code :show :result :show}}

#_(clerk/row
  (for [mf [#'clojure.math/sin #'clojure.math/cos #'clojure.math/tan]]
    (let [f (juxt identity mf)
          data (map f (range -10 10 0.1))]
      (clerk/plotly
        {:data [{:x (map first data) :y (map peek data)}]
         :layout {:title (str mf) :width 400}}))))

#_{:nextjournal.clerk/visibility {:code :hide :result :hide}}

(def graph-viewer
  {:pred :graph
   :transform-fn
   (clerk/update-val
     (fn [m]
       (let [path (str "tmp/graph-" (hash m) ".png")
             opts (get m :opts {})]
         (io/make-parents (io/file path))
         (gv/render-graph (:graph m) (assoc opts :filename path))
         (clerk/col {::clerk/width :wide}
           (ImageIO/read (io/file path))))))})

(clerk/add-viewers! [graph-viewer])

(defn render-graph [graph & {:keys [] :as opts}]
  {:graph
   (merge
     {:flags #{:directed}
      :default-attributes {:graph {:rankdir "LR"}}}
     graph)
   :opts opts})

{:nextjournal.clerk/visibility {:code :show :result :show}}

(render-graph {:nodes ["a" "b" "c"]})

(defn draw* [{:keys [v op children]} points-to nodes edges]
  (let [add-node (comp (filter some?) (map str))
        add-edge (comp (filter #(every? some? %)) (map #(mapv str %)))
        nodes (into nodes add-node [op v])
        edges (into edges add-edge [[op v] [v points-to]])]
    (reduce
      (fn [[nodes edges] child] (draw* child op nodes edges))
      [nodes edges]
      children)))

(defn draw [v]
  (let [[n e] (draw* v nil #{} #{})]
    (render-graph {:nodes n :edges e})))

(def result (+ (* (value 2.0) (value 5.0)) (value 3.0)))
(draw result)



(def a 2)
(def b -3)
(def c 10)
^{:nextjournal.clerk/no-cache true}
(def d '(+ (* a b) c))

(def nodes (atom #{}))
(defn traverse [op & d]
  (doseq [])
  (let [[op & args] d
        args (doall
               (for [a args]
                 (if (list? a)
                   (traverse a)
                   (let [a+ (eval #p a)
                         a (str a "=" a+)]
                     (swap! nodes conj a)
                     a+))))
        ret (eval (cons op args))]
    (swap! nodes c)
    )

