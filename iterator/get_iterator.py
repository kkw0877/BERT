import tensorflow as tf


def pretrain_iterator(file_name,
                      seq_len,
                      max_predictions_per_seq,
                      batch_size):
    dataset = tf.data.TFRecordDataset(filenames=[file_name])

    feature_description = {
        'masked_lm_ids': tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        'masked_lm_weights': tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        'next_sentence_labels': tf.io.FixedLenFeature([1], tf.int64),
        'input_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
        'input_mask': tf.io.FixedLenFeature([seq_len], tf.int64),
        'masked_lm_positions': tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
    }

    def _parse_function(example_proto):
        return tf.io.parse_example(example_proto, feature_description)

    parsed_dataset = dataset.map(_parse_function)
    # [] means "unused labels"
    parsed_dataset = parsed_dataset.map(lambda record: (record, []))
    parsed_dataset = parsed_dataset.batch(batch_size)
    return parsed_dataset

def squad_iterator(file_name,
                   seq_length,
                   batch_size):
    
    dataset = tf.data.TFRecordDataset(filenames=[file_name])
    
    input_features = {
        'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64)}
    label_features = {
        'start_positions': tf.io.FixedLenFeature([], tf.int64),
        'end_positions': tf.io.FixedLenFeature([], tf.int64)}

    def _parse_function(example_proto):
        inputs = tf.io.parse_example(example_proto, input_features)
        labels = tf.io.parse_example(example_proto, label_features)
        labels = tf.stack([labels['start_positions'], labels['end_positions']], axis=0)
        return inputs, labels

    dataset = dataset.batch(batch_size)
    parsed_dataset = dataset.map(lambda record : _parse_function(record))
    return parsed_dataset


