import tensorflow as tf

class Embedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 initializer,
                 **kwargs):
    
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.initializer = initializer

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name='embedding',
            shape=(self.vocab_size, self.embedding_dim),
            initializer=self.initializer,
            dtype=tf.float32)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embedding, inputs)
