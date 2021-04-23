import tensorflow as tf

from model import layers

class SquadModel(tf.keras.Model):
    def __init__(self, 
                 core_model, 
                 seq_len, 
                 kernel_initializer):

        # inputs for core model
        input_ids = tf.keras.Input(
            shape=(seq_len,), dtype=tf.int32, name='input_ids') 
        input_mask = tf.keras.Input(
            shape=(seq_len,), dtype=tf.int32, name='input_mask')
        segment_ids = tf.keras.Input(
            shape=(seq_len,), dtype=tf.int32, name='segment_ids')

        cls_outputs, enc_outputs = core_model([input_ids, input_mask, segment_ids])
        
        logits_layer = layers.LogitsLayer(kernel_initializer)
        logits = logits_layer(enc_outputs)
        
        inputs = {'input_ids':input_ids, 
                  'input_mask':input_mask, 
                  'segment_ids':segment_ids}

        outputs = logits

        super(SquadModel, self).__init__(inputs=inputs, outputs=outputs)
