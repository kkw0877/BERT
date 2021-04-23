class BertConfig(object):
  def __init__(self,
               vocab_size=28996,
               hidden_size=128, 
               max_sequence_len=256,
               segment_size=2, 
               dropout_rate=0.1,
               num_blocks=6,
               num_heads=8, 
               ffn_dim=56, 
               kernel_initializer='glorot_uniform', 
               activation='gelu',
               max_preds_per_seq=76,
               seq_len=256):
    
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.max_sequence_len = max_sequence_len
    self.segment_size = segment_size
    self.dropout_rate = dropout_rate
    self.num_blocks = num_blocks
    self.num_heads = num_heads
    self.ffn_dim = ffn_dim
    self.kernel_initializer = kernel_initializer
    self.activation = activation
    self.max_preds_per_seq = max_preds_per_seq
    self.seq_len = seq_len