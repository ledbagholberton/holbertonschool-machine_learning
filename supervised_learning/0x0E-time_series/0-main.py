#!/usr/bin/env python3
"""
Data preprocessing:
Visually to predict data we can cut data from 2017
"""
import pandas as pd
import tensorflow as tf
multivariate_data = __import__('1-multivariate').multivariate_data
plot_train_history = __import__('2-plots').plot_train_history
show_plot = __import__('2-plots').show_plot

past_history = 24
future_target = 0
TRAIN_SPLIT = 1300000
STEP = 1
dataset = multivariate_data(TRAIN_SPLIT)
x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 2], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 2],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)
BUFFER_SIZE = 18768
BATCH_SIZE = 256
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(24, input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))
single_step_model.summary()
single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mse')
EPOCHS = 30
EVALUATION_INTERVAL = 500
single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)
plot_train_history(single_step_history,
                   'Single Step Training and validation loss')
for x, y in val_data_single.take(10):
  plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                   single_step_model.predict(x)[0]], 1,
                   'Single Step Prediction')
  plot.show()
