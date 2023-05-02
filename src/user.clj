(ns user)

(defmacro exprs
  "Take a let-shaped values vector of binding + expr pairs and return a map
  of expression data with gradients initialized to zero."
  [values]
  `(let [~@values ~@[]]
     (array-map
       ~@(mapcat
           (fn [[k expr]]
             [(list `quote k) {:expr (list `quote expr) :data k :grad 0.0}])
           (partition-all 2 values)))))

(defn children
  "Get the children for an expression node, if any."
  [{:keys [expr]}]
  (when (list? expr)
    (rest expr)))

(defmulti backwards*
  "Dispatch on the expression operation (e.g. `*` or `+`) and back-propagate
  the gradient of the output node to its input nodes by updating `env`."
  (fn [env {:keys [expr]}]
    (when (list? expr)
      (first expr))))

(defmethod backwards* :default [env _] env)

; Derivative for addition of two values.
(defmethod backwards* '+
  [env {:keys [grad] :as node}]
  (let [[self other] (children node)]
    (-> env
        (update-in [self :grad] (fnil + 0.0) grad)
        (update-in [other :grad] (fnil + 0.0) grad))))

; Derivative for multiplication of two values.
(defmethod backwards* '*
  [env {:keys [grad] :as node}]
  (let [[self other] (children node)
        + (fnil + 0.0)]
    ; Using `as->` so that each line has the newest version of `env`
    (as-> env env
      (update-in env [self :grad] + (* (get-in env [other :data]) grad))
      (update-in env [other :grad] + (* (get-in env [self :data]) grad)))))

(defn backwards
  "Given an environment map of expression nodes, back-propagate the gradient
  starting at the node with the given `name`."
  [env name]
  (let [node (get env name)
        env (backwards* env node)]
    (reduce backwards env (children node))))

(comment
  ;; For example, using the test case from https://youtu.be/VMj-3S1tku0?t=5178

  (let [; define the expressions
        env (exprs [a -2
                    b 3
                    d (* a b)
                    e (+ a b)
                    f (* d e)])
        ; set the gradient of `f` to 1.0 to kick things off
        env (assoc-in env ['f :grad] 1.0)]
    ; back-propagate the gradient
    (backwards env 'f))
  ;=>
  '{a {:expr -2, :data -2, :grad -3.0},
    b {:expr 3, :data 3, :grad -8.0},
    d {:expr (* a b), :data -6, :grad 1.0},
    e {:expr (+ a b), :data 1, :grad -6.0},
    f {:expr (* d e), :data -6, :grad 1.0}}
  )
