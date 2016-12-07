```
$ git clone https://github.com/levinas/Benchmarks
$ cd Benchmarks

$ python -m benchmarks.P1B2.p1b2_baseline

Using TensorFlow backend.

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
dense_1 (Dense)                  (None, 256)           2559744     dense_input_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 256)           65792       dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            2570        dense_2[0][0]
====================================================================================================
Total params: 2628106
____________________________________________________________________________________________________

Train on 2400 samples, validate on 600 samples
Epoch 1/1
2400/2400 [==============================] - 4s - loss: 2.6116 - acc: 0.3538 - val_loss: 2.0785 - val_acc: 0.4100

{'categorical_crossentropy': 1.9021, 'leaderboard_metric': 'categorical_crossentropy', 'accuracy': 0.3237}

Submitting to leaderboard...

```

