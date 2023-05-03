(ns com.mjdowney.micrograd
  "autograd engine for scalar values and operations in the style of

    https://github.com/karpathy/micrograd

  but with immutable values.")

; Wrap a scalar `data` with some additional context
(defrecord Value [data grad children op op-deriv])

(defn value
  "Stick the scalar `n` in a `Value` record."
  [n]
  (map->Value {:data n :grad 0.0}))

(defn- operation
  "Helper for implementing math.

  Apply `op` to each of the named values in `env`

    (calling `(op (get env vn) ...)`)

  and return a new `Value` with the result."
  [env op op-deriv vn & vns]
  (let [args (cons vn vns)
        data (map (comp :data env) args)]
    (->Value (apply op data) 0.0 args op op-deriv)))

; Define some math. Derivative functions take the gradient of the output
; and the value names for the children of the operation, and return a map of
; child value name -> gradient.

(defn- mul-deriv [env og [vn1 vn2]]
  ; look up the child values in the environment
  {vn1 (-> env vn2 :data (* og))
   vn2 (-> env vn1 :data (* og))})

(defn- add-deriv [_env og [v0 v1]] {v0 og v1 og})

(defn mul "Multiplication." [env v0 v1] (operation env #'* #'mul-deriv v0 v1))
(defn add "Addition." [env v0 v1] (operation env #'+ #'add-deriv v0 v1))

(defn- add-gradient
  "Update the :grad for the value named `vname` by summing in the `delta`."
  [env vname delta]
  (update-in env [vname :grad] + delta))

(defn- step-back [env name]
  (let [v (get env name)]
    ; If there's a backwards op, call it on the children and update the env
    ; with the resulting changes, otherwise return env unchanged
    (if-let [op' (:op-deriv v)]
      (reduce-kv add-gradient env (op' env (:grad v) (:children v)))
      env)))

(defn backwards
  "Set the gradient of the value named `vn` to 1.0, and then backpropagate
  through the graph to update the gradients of all other values."
  [env vn]
  (loop [q [vn]
         env (assoc-in env [vn :grad] 1.0)]
    (if-let [nxt (first q)]
      (recur (concat (rest q) (-> env nxt :children)) (step-back env nxt))
      env)))

^:rct/test
(comment
  ;; For example, using the test case from https://youtu.be/VMj-3S1tku0?t=5178

  ; Define a computational graph with the values and operations
  (def e
    (as-> {} env
      (assoc env :a (value -2))
      (assoc env :b (value 3))
      (assoc env :d (mul env :a :b))
      (assoc env :e (add env :a :b))
      (assoc env :f (mul env :d :e))))

  e
  ; {:a #micrograd.Value{:data -2, :grad 0.0, :children nil, :op nil, :op-deriv nil},
  ;  :b #micrograd.Value{:data 3, :grad 0.0, :children nil, :op nil, :op-deriv nil},
  ;  :d #micrograd.Value{:data -6, :grad 0.0, :children (:a :b), :op #'clojure.core/*, :op-deriv #'micrograd/mul-deriv},
  ;  :e #micrograd.Value{:data 1, :grad 0.0, :children (:a :b), :op #'clojure.core/+, :op-deriv #'micrograd/add-deriv},
  ;  :f #micrograd.Value{:data -6, :grad 0.0, :children (:d :e), :op #'clojure.core/*, :op-deriv #'micrograd/mul-deriv}}


  ; Take the gradients with respect to :f
  (backwards e :f)
  ; {:a #micrograd.Value{:data -2, :grad -3.0, :children nil, :op nil, :op-deriv nil},
  ;  :b #micrograd.Value{:data 3, :grad -8.0, :children nil, :op nil, :op-deriv nil},
  ;  :d #micrograd.Value{:data -6, :grad 1.0, :children (:a :b), :op #'clojure.core/*, :op-deriv #'micrograd/mul-deriv},
  ;  :e #micrograd.Value{:data 1, :grad -6.0, :children (:a :b), :op #'clojure.core/+, :op-deriv #'micrograd/add-deriv},
  ;  :f #micrograd.Value{:data -6, :grad 1.0, :children (:d :e), :op #'clojure.core/*, :op-deriv #'micrograd/mul-deriv}}

  (update-vals (backwards e :f) :grad)
  ;=> {:a -3.0, :b -8.0, :d 1.0, :e -6.0, :f 1.0}
  )
