(ns com.mjdowney.clerk
  "Entrypoint for Clerk notebook server."
  (:require [nextjournal.clerk :as clerk]))

(defn serve! [_]
  (clerk/serve!
    {:browse? true
     :watch-paths ["src"]})
  (clerk/show! "src/com/mjdowney/micrograd.clj"))

(comment
  ; Or just eval this from a REPL session :)
  (serve! {})
  (clerk/halt!)
  )
