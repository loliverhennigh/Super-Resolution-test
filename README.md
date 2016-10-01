# Super-Resolution-test
A test to see if I can get the ops for super resolution. Right now to run just do
```
python main.py
```
Three images will appear in the `images` directory. The true high res image, the true low res image, and the predicted high res image from the low res image. Its all from the bouncing ball data set. deconvs are used in this network right now but later Ill have rotating conv.

## Todos
Get the rotating conv op working. I have a `copy_op.cc` working from [here](http://stackoverflow.com/questions/36204225/tensorflow-custom-op-gradient). Now I just need to mess with the for loop doing the copying.
