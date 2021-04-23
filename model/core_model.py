import tensorflow as tf

from model import layers

class BertCoreModel(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_sequence_len,
                 segment_size,
                 dropout_rate,
                 num_blocks,
                 num_heads,
                 ffn_dim,
                 kernel_initializer,
                 activation,
                 **kwargs):

        activation = tf.keras.activations.get(activation)
        initializer = tf.keras.initializers.get(kernel_initializer)

        # inputs for inner model
        input_ids = tf.keras.Input(
            shape=(None,), dtype=tf.int32, name='input_ids')
        input_mask = tf.keras.Input(
            shape=(None,), dtype=tf.int32, name='input_mask')
        segment_ids = tf.keras.Input(
            shape=(None,), dtype=tf.int32, name='segment_ids')

        # word embedding, positional encoding, segment embedding
        word_embedding_layer = layers.Embedding(
            vocab_size, hidden_size, initializer, **kwargs)
        word_embedding = word_embedding_layer(input_ids)

        positional_encoding_layer = layers.PositionalEncoding(
            max_sequence_len, hidden_size)
        positional_encoding = positional_encoding_layer(input_ids)

        segment_embedding_layer = layers.Embedding(
            segment_size, hidden_size, initializer, **kwargs)
        segment_embedding = segment_embedding_layer(segment_ids)


        inputs = tf.keras.layers.add(
            [word_embedding, positional_encoding, segment_embedding])
        inputs = tf.keras.layers.Dropout(dropout_rate)(inputs)

        # forward pass
        enc_outputs = inputs
        for _ in range(num_blocks):
            encoder_block = layers.EncoderBlock(
                num_heads,
                initializer,
                activation,
                ffn_dim,
                dropout_rate)

            enc_outputs = encoder_block([enc_outputs, input_mask])

        # dense layer for next sentence prediction
        cls_layer = tf.keras.layers.Dense(
            units=hidden_size,
            kernel_initializer=initializer) # d_model(=embedding_dim), initializer
        cls_outputs = cls_layer(enc_outputs[:, 0, :])

        super(BertCoreModel, self).__init__(
            inputs=[input_ids, input_mask, segment_ids],
            outputs=[cls_outputs, enc_outputs],
            **kwargs)
