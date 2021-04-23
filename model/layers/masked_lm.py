import tensorflow as tf

class MaskedLM(tf.keras.layers.Layer):
  def __init__(self, seq_length, embedding_dim, vocab_size, kernel_initializer, **kwargs):
    super(MaskedLM, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.kernel_initializer = kernel_initializer
    self.seq_length = seq_length
    self.embedding_dim = embedding_dim

  def call(self, inputs, positions):
    extracted_outputs = self._extract_tensor(inputs, positions)
    masked_lm_logits = self.dense(extracted_outputs)
    return masked_lm_logits

  def build(self, input_shape):
    self.dense = tf.keras.layers.Dense(self.vocab_size,
      kernel_initializer=self.kernel_initializer)
    
    super(MaskedLM, self).build(input_shape)

  def _extract_tensor(self, seq_outputs, masked_lm_positions):
    batch_size = tf.shape(masked_lm_positions)[0]
    seq_length = self.seq_length
    embedding_dim = self.embedding_dim

    # [batch_size*seq_length, embedding_dim]
    seq_outputs = tf.reshape(seq_outputs, [-1, embedding_dim]) 

    temp = tf.reshape(tf.range(batch_size)*seq_length, (-1, 1))
    temp_positions = tf.reshape(temp+masked_lm_positions, [-1])
    
    extracted_outputs = tf.gather(seq_outputs, temp_positions)
    # [batch_size, max_preds_per_seq, embedding_dim]
    extracted_outputs = tf.reshape(extracted_outputs, 
      [batch_size, -1, embedding_dim])

    return extracted_outputs
