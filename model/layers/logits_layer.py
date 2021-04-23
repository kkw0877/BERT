import tensorflow as tf

class LogitsLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_initializer, **kwargs): 
        super(LogitsLayer, self).__init__(**kwargs)
        self.kernel_initializer = kernel_initializer
    
    def call(self, inputs):
        enc_outputs = inputs
        logits = self.dense(enc_outputs)
        return self._transpose_tensor(logits)
    
    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(2,
            kernel_initializer=self.kernel_initializer)
        
        super(LogitsLayer, self).build(input_shape)
        
    def _transpose_tensor(self, logits):
        return tf.transpose(logits, [2, 0, 1])
