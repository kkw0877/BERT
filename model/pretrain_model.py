import tensorflow as tf

from model import layers

class PretrainModel(tf.keras.Model):
    def __init__(self,
                 core_model,
                 seq_len,
                 max_preds_per_seq,
                 kernel_initializer,
                 hidden_size,
                 vocab_size):

        # inputs for core model
        input_ids = tf.keras.Input(
            shape=(seq_len,), dtype=tf.int32, name='input_ids')
        input_mask = tf.keras.Input(
            shape=(seq_len,), dtype=tf.int32, name='input_mask')
        segment_ids = tf.keras.Input(
            shape=(seq_len,), dtype=tf.int32, name='segment_ids')

        # 1. masked lm
        # masked_lm_ids, masked_lm_positions, masked_lm_weights
        masked_lm_ids = tf.keras.Input(
            (max_preds_per_seq,), dtype=tf.int32, name='masked_lm_ids')
        masked_lm_positions = tf.keras.Input(
            (max_preds_per_seq,), dtype=tf.int32, name='masked_lm_positions')
        masked_lm_weights = tf.keras.Input(
            (max_preds_per_seq,), dtype=tf.float32, name='masked_lm_weights')

        # 2. next sentence prediction
        # next_sentence_labels
        next_sentence_labels = tf.keras.Input(
            (1,), dtype=tf.int32, name='next_sentence_labels')

        cls_outputs, enc_outputs = core_model([input_ids, input_mask, segment_ids])

        # next sentence prediction layer (cls_outputs) -> nsp_logits
        kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        nsp_dense = tf.keras.layers.Dense(
            units=2,
            kernel_initializer=kernel_initializer)
        nsp_logits = nsp_dense(cls_outputs)

        # masked lm layer (enc_outputs, masked_lm_positions) -> masked_lm_logits
        masked_lm = layers.MaskedLM(
            seq_len, hidden_size, vocab_size, kernel_initializer)
        masked_lm_logits = masked_lm(enc_outputs, masked_lm_positions)

        # LossAndMetricLayer
        # inputs 1) nsp_logits, next_sentence_labels
        # inputs 2) masked_lm_logits, masked_lm_ids, masked_lm_weights
        loss_metric_layer = LossAndMetricLayer()
        loss = loss_metric_layer(next_sentence_labels,
                                 nsp_logits,
                                 masked_lm_ids,
                                 masked_lm_logits,
                                 masked_lm_weights)

        inputs = {'input_ids':input_ids,
                  'input_mask':input_mask,
                  'segment_ids':segment_ids,
                  'masked_lm_ids':masked_lm_ids,
                  'masked_lm_positions':masked_lm_positions,
                  'masked_lm_weights':masked_lm_weights,
                  'next_sentence_labels':next_sentence_labels}

        outputs = loss

        super(PretrainModel, self).__init__(inputs=inputs, outputs=outputs)

class LossAndMetricLayer(tf.keras.layers.Layer):
    def call(self, nsp_labels, nsp_logits, masked_lm_ids, masked_lm_logits, masked_lm_weights):
        total_loss, nsp_loss, masked_lm_loss = self._get_loss(
            nsp_labels, nsp_logits, masked_lm_ids, masked_lm_logits, masked_lm_weights)

        self._add_metric(
            nsp_loss, masked_lm_loss, nsp_labels, nsp_logits,
            masked_lm_ids, masked_lm_logits, masked_lm_weights)

        return total_loss

    def _get_loss(self, nsp_labels, nsp_logits, masked_lm_ids, masked_lm_logits, masked_lm_weights):
        nsp_loss = tf.keras.losses.sparse_categorical_crossentropy(
              nsp_labels, nsp_logits, from_logits=True)
        nsp_loss = tf.reduce_mean(nsp_loss)

        pred_loss = tf.keras.losses.sparse_categorical_crossentropy(
            masked_lm_ids, masked_lm_logits, from_logits=True)

        # exclude pads for accurate calculation
        masked_lm_numerator = tf.reduce_sum(pred_loss * masked_lm_weights)
        masked_lm_denominator = tf.reduce_sum(masked_lm_weights)
        masked_lm_loss = tf.math.divide_no_nan(masked_lm_numerator, masked_lm_denominator)

        total_loss = nsp_loss + masked_lm_loss
        return total_loss, nsp_loss, masked_lm_loss

    def _add_metric(self, nsp_loss, masked_lm_loss, nsp_labels, nsp_logits, masked_lm_ids, masked_lm_logits, masked_lm_weights):
        # next sentence metrics
        # 1. next sentence accuracy
        nsp_acc = tf.keras.metrics.sparse_categorical_accuracy(
            nsp_labels, nsp_logits)
        self.add_metric(
            nsp_acc,
            name='next_sentence_prediction_accuracy',
            aggregation='mean')

        # 2. next sentence loss mean
        self.add_metric(
            nsp_loss,
            name='next_sentence_prediction_loss',
            aggregation='mean')

        # masked lm metrics
        # 1. masked lm accuracy
        pred_acc = tf.keras.metrics.sparse_categorical_accuracy(
            masked_lm_ids, masked_lm_logits)
        masked_lm_numerator = tf.reduce_sum(pred_acc * masked_lm_weights)
        masked_lm_denominator = tf.reduce_sum(masked_lm_weights) + 1e-4
        masked_lm_acc = masked_lm_numerator / masked_lm_denominator
        self.add_metric(
            masked_lm_acc,
            name='masked_lm_accuracy',
            aggregation='mean')

        # 2. masked lm loss mean
        self.add_metric(
            masked_lm_loss,
            name='masked_lm_loss',
            aggregation='mean')
