import tensorflow as tf

import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, max_seq_length, d_model):
    super(PositionalEncoding, self).__init__()
    self.max_seq_length = max_seq_length
    self.d_model = d_model

  def build(self, input_shape):
    row = np.arange(self.max_seq_length).reshape((-1, 1))
    col = 2*(np.arange(self.d_model)//2)
    col = col.reshape((1, -1))

    pos_enc = row / np.power(10000, col/np.float(self.d_model))
    pos_enc[:, ::2] = np.sin(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
    pos_enc = pos_enc[np.newaxis, ...]
    pos_enc = tf.cast(pos_enc, dtype=tf.float32)

    self.pos_enc = pos_enc

  def call(self, inputs):
    seq_length = inputs.shape[1]
    return self.pos_enc[:, :seq_length, :]
    
