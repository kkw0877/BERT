import tensorflow as tf

import train
from model.core_model import BertCoreModel
from model.pretrain_model import PretrainModel
from model.squad_model import SquadModel

def load_squad_model(vocab_size,
                     hidden_size, 
                     max_sequence_len, 
                     segment_size, 
                     dropout_rate,
                     num_blocks,
                     num_heads,
                     ffn_dim, 
                     kernel_initializer, 
                     activation,
                     seq_len,
                     core_dir):
    
    inner_model = BertCoreModel(
        vocab_size,
        hidden_size, 
        max_sequence_len, 
        segment_size, 
        dropout_rate,
        num_blocks,
        num_heads,
        ffn_dim, 
        kernel_initializer, 
        activation)
    
    if tf.train.latest_checkpoint(core_dir):
        core_ckpt = tf.train.Checkpoint(inner_model=inner_model)
        core_ckpt.restore(tf.train.latest_checkpoint(core_dir))
        print('Restored from {}'.format(tf.train.latest_checkpoint(core_dir)))
    else:
        print('Initializing from scratch.')
    
    squad_model = SquadModel(inner_model,
                             seq_len,
                             kernel_initializer)
              
    return squad_model, inner_model

def run_squad(train_iterator, 
              eval_iterator, 
              model, 
              loss_fn, 
              metric_fn, 
              step, 
              model_dir, 
              inner_model_dir, 
              summary_dir, 
              train_summary_interval,
              eval_summary_interval,
              epochs):
    
    train.customized_training_loop(train_iterator,
                                   eval_iterator,
                                   model,
                                   loss_fn,
                                   metric_fn,
                                   step,
                                   model_dir,
                                   inner_model_dir,
                                   summary_dir,
                                   train_summary_interval,
                                   eval_summary_interval,
                                   epochs)



def load_pretrain_model(vocab_size,
                        hidden_size, 
                        max_sequence_len, 
                        segment_size, 
                        dropout_rate,
                        num_blocks,
                        num_heads,
                        ffn_dim, 
                        kernel_initializer, 
                        activation,
                        seq_len,
                        max_preds_per_seq):
  
    core_model = BertCoreModel(
        vocab_size,
        hidden_size, 
        max_sequence_len, 
        segment_size, 
        dropout_rate,
        num_blocks,
        num_heads,
        ffn_dim, 
        kernel_initializer, 
        activation)
    
    pretrain_model = PretrainModel(
        core_model,
        seq_len,
        max_preds_per_seq,
        kernel_initializer,
        hidden_size,
        vocab_size)          
    
    return pretrain_model, core_model

def run_pretrain(train_iterator, 
                 eval_iterator, 
                 model, 
                 loss_fn, 
                 metric_fn, 
                 step, 
                 model_dir, 
                 inner_model_dir, 
                 summary_dir, 
                 train_summary_interval,
                 eval_summary_interval,
                 epochs):

    train.customized_training_loop(train_iterator,
                                   eval_iterator,
                                   model,
                                   loss_fn,
                                   metric_fn,
                                   step,
                                   model_dir,
                                   inner_model_dir,
                                   summary_dir,
                                   train_summary_interval,
                                   eval_summary_interval,
                                   epochs)
