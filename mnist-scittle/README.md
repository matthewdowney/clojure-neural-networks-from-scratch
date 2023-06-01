# mnist-scittle

A small network (one layer of 30 neurons), trained on the MNIST digits dataset,
embedded in a widget using [Scittle](https://github.com/babashka/scittle).

Run locally with `bb dev`.

## Structure
The `canvas.js` file is copied / refactored from 
[ccom.ucsd.edu/~cdeotte/programs/MNIST](http://www.ccom.ucsd.edu/~cdeotte/programs/MNIST.html). 
It uses an HTML canvas to get an input vector of pixels, and is significantly 
more work than the actual digit recognition!

The `mnist.cljs` file does the actual digit recognition and handles the UI 
interactions.
