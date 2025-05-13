from sat.model import BaseMixin
from sat.ops import memory_efficient_attention
from sat.mpu.utils import split_tensor_along_last_dim


class MemoryEfficientAttentionMixin(BaseMixin):
    ''' Flash Attention
        Fp32 is identical to the original implementation, fp16 has differences.
    '''
    def __init__(self):
        super().__init__()
    
    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        if not scaling_attention_score:
            scale = 1.
        else:
            scale = None
        
        if attention_dropout is not None:
            attention_dropout = attention_dropout.p     
        else:
            attention_dropout = 0.0    

        return memory_efficient_attention(query_layer.transpose(1, 2), key_layer.transpose(1, 2), value_layer.transpose(1, 2), mask=attention_mask, scale=scale, attention_dropout=attention_dropout).transpose(1, 2)
    
class TransposedMemoryEfficientAttentionMixin(BaseMixin):
    ''' 
        Avoid permute by keeping [B L H D] format, might not compatible with p-tuning etc.
    '''
    def __init__(self):
        super().__init__()
    
    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        if not scaling_attention_score:
            scale = 1.
        else:
            scale = None
        
        if attention_dropout is not None:
            attention_dropout = attention_dropout.p     
        else:
            attention_dropout = 0.0

        return memory_efficient_attention(query_layer, key_layer, value_layer, mask=attention_mask, scale=scale, attention_dropout=attention_dropout)
    
    def attention_forward(self, hidden_states, mask, **kw_args):
        attention_fn = self.attention_fn
        self = self.transformer.layers[kw_args['layer_id']].attention
        if 'attention_fn' in self.hooks and self.hooks['attention_fn'] != attention_fn:
            from sat.helpers import print_rank0
            print_rank0('Dangerous! Memory efficient attention uses B L H D format, different from the default attention! You should not change attention_fn except you already noticed it.', level='warning')

        mixed_raw_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = self.attention_dropout if self.training else None

        query_layer = mixed_query_layer.view(mixed_query_layer.shape[0], mixed_query_layer.shape[1], self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        key_layer = mixed_key_layer.view(mixed_key_layer.shape[0], mixed_key_layer.shape[1], self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        value_layer = mixed_value_layer.view(mixed_value_layer.shape[0], mixed_value_layer.shape[1], self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output