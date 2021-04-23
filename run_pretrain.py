# main.py
from absl import flags
from absl import app

import iterator
import bert_model
import optimizer
import losses
import bert_config

# train_input_fn
flags.DEFINE_string('train_file_name', './train_small_tf_examples.tfrecord', '.')
flags.DEFINE_string('eval_file_name', './valid_small_tf_examples.tfrecord', '.')
flags.DEFINE_integer('seq_len', 256, '.')
flags.DEFINE_integer('max_predictions_per_seq', 76, '.')
flags.DEFINE_integer('batch_size', 16, '.')

# optimizer
flags.DEFINE_float('init_lr', 1e-4, '.')
#flags.DEFINE_integer('data_len', 10359941, '.') # train_tf_examples.tfrecord
flags.DEFINE_integer('data_len', 1093331, '.') # train_small_tf_examples.tfrecord
flags.DEFINE_float('beta_1', 0.9, '.')
flags.DEFINE_float('beta_2', 0.999, '.')

# procedure for pretrain model
flags.DEFINE_integer('step', 1, '.')
flags.DEFINE_string('model_dir', 'out_model', '.')
flags.DEFINE_string('inner_model_dir', 'inner_model', '.')
flags.DEFINE_string('summary_dir', 'bert_summary', '.')
flags.DEFINE_integer('train_summary_interval', 50, '.')
flags.DEFINE_integer('eval_summary_interval', 50, '.')
flags.DEFINE_integer('epochs', 10, '.')

FLAGS = flags.FLAGS

def get_train_iterator(file_name,
                       seq_len,
                       max_predictions_per_seq,
                       batch_size):
    
    train_iter = iterator.pretrain_iterator(
        file_name=file_name,
        seq_len=seq_len,
        max_predictions_per_seq=max_predictions_per_seq,
        batch_size=batch_size)
  
    return train_iter

def get_eval_iterator(file_name,
                      seq_len,
                      max_predictions_per_seq,
                      batch_size):
    
    eval_iter = iterator.pretrain_iterator(
        file_name=file_name,
        seq_len=seq_len,
        max_predictions_per_seq=max_predictions_per_seq,
        batch_size=batch_size)
  
    return eval_iter

def get_loss_fn():
    return losses.get_bert_pretrain_loss_fn

def get_pretrain_model(config):
    pretrain_model, core_model = bert_model.load_pretrain_model(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        max_sequence_len=config.max_sequence_len,
        segment_size=config.segment_size,
        dropout_rate=config.dropout_rate,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        kernel_initializer=config.kernel_initializer,
        activation=config.activation,
        seq_len=config.seq_len,
        max_preds_per_seq=config.max_preds_per_seq)

    steps_per_epoch = FLAGS.data_len // FLAGS.batch_size
    num_train_steps = steps_per_epoch * FLAGS.epochs
    warmup_steps = num_train_steps // 100

    pretrain_model.optimizer = optimizer.get_optimizer(
        init_lr=FLAGS.init_lr,
        warmup_steps=warmup_steps,
        num_train_steps=num_train_steps,
        beta_1=FLAGS.beta_1,
        beta_2=FLAGS.beta_2)
  
    return pretrain_model, core_model

def run_pretrain_model(config):
    # train_input_fn
    train_iterator = get_train_iterator(
        file_name=FLAGS.train_file_name,
        seq_len=FLAGS.seq_len,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        batch_size=FLAGS.batch_size)
  
    # eval_input_fn
    eval_iterator = get_eval_iterator(
        file_name=FLAGS.eval_file_name,
        seq_len=FLAGS.seq_len,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        batch_size=FLAGS.batch_size)
  
    # model_fn
    pretrain_model, core_model = get_pretrain_model(config)
    model = (pretrain_model, core_model)

    # loss_fn
    loss_fn = get_loss_fn()

    # start training
    bert_model.run_pretrain(
        train_iterator=train_iterator,
        eval_iterator=eval_iterator,
        model=model,
        loss_fn=loss_fn,
        metric_fn=None,
        step=FLAGS.step,
        model_dir=FLAGS.model_dir,
        inner_model_dir=FLAGS.inner_model_dir,
        summary_dir=FLAGS.summary_dir,
        train_summary_interval=FLAGS.train_summary_interval,
        eval_summary_interval=FLAGS.eval_summary_interval,
        epochs=FLAGS.epochs)

def main(_):
    # initialize config
    config = bert_config.BertConfig()

    run_pretrain_model(config)

if __name__ == '__main__':
    app.run(main)