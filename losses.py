import tensorflow as tf

def get_bert_pretrain_loss_fn(unused_labels, losses, **kwargs):
    return tf.reduce_mean(losses)
    # return losses

def get_bert_squad_loss_fn(labels, logits):
    loss = tf.keras.metrics.sparse_categorical_crossentropy(
        labels, logits)
    loss = tf.reduce_mean(loss)

    return loss
