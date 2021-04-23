import tensorflow as tf

class EncoderBlock(tf.keras.layers.Layer):
  def __init__(self, num_heads, initializer, activation, ffn, rate, **kwargs):
    super(EncoderBlock, self).__init__(**kwargs)
    self.num_heads = num_heads
    self.initializer = tf.keras.initializers.get(initializer)
    self.activation = tf.keras.activations.get(activation)
    self.ffn = ffn
    self.dropout_rate = rate

  def build(self, input_shape):
    batch_size, seq_len, d_model = input_shape[0]
    if d_model % self.num_heads != 0:
      raise ValueError("Can't divide d_model by num_heads")
    
    key_dim = d_model // self.num_heads

    self.mha = tf.keras.layers.MultiHeadAttention(
        self.num_heads, key_dim)
    self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)
    self.layer_norm1 = tf.keras.layers.LayerNormalization()

    self.inner_dense = tf.keras.layers.Dense(
        units=self.ffn, 
        activation=self.activation, 
        kernel_initializer=self.initializer)
    self.drop2 = tf.keras.layers.Dropout(self.dropout_rate)

    self.outer_dense = tf.keras.layers.Dense(
        units=d_model,
        kernel_initializer=self.initializer)
    self.drop3 = tf.keras.layers.Dropout(self.dropout_rate)

    self.layer_norm2 = tf.keras.layers.LayerNormalization()

    super(EncoderBlock, self).build(input_shape)

  def call(self, inputs):
    if not isinstance(inputs, list):
      raise ValueError('Inputs must be [inputs, input_mask]')

    inputs, input_mask = inputs
    attn_mask = self._compute_mask(inputs, input_mask)

    attn_outputs = self.mha(
        query=inputs, value=inputs, attention_mask=attn_mask)
    attn_outputs = self.drop1(attn_outputs)
    attn_outputs = self.layer_norm1(attn_outputs + inputs)

    layer_outputs = self.inner_dense(attn_outputs)
    layer_outputs = self.drop2(layer_outputs)
    layer_outputs = self.outer_dense(layer_outputs)
    layer_outputs = self.drop3(layer_outputs)
    layer_outputs = self.layer_norm2(attn_outputs + layer_outputs)

    return layer_outputs

  def _compute_mask(self, inputs, input_mask):
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    seq_len = inputs_shape[1]
    temp = tf.ones((batch_size, seq_len, 1), dtype=input_mask.dtype)
    attn_mask = tf.expand_dims(input_mask, axis=1) * temp
    return attn_mask
