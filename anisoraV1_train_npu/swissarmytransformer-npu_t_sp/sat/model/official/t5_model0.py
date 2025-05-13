import math
import torch
import torch.nn.functional as F
from sat.model.mixins import BaseMixin
from sat.model.encoder_decoder_model import EncoderDecoderModel
from sat.model.base_model import non_conflict
from sat.mpu import get_model_parallel_world_size
from sat.model.transformer import standard_attention, SelfAttention, CrossAttention, MLP
from sat.mpu.mappings import copy_to_model_parallel_region
from sat.mpu.utils import divide, split_tensor_along_last_dim, unscaled_init_method
from sat.mpu.layers import ColumnParallelLinear, VocabParallelEmbedding


class T5PositionEmbeddingMixin(BaseMixin):
    def position_embedding_forward(self, position_ids, **kw_args):
        return None


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 or bfloat16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        elif self.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        return self.weight * hidden_states


class T5AttentionMixin(BaseMixin):
    def __init__(self, relative_attention_num_buckets, num_attention_heads, is_decoder=False):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        world_size = get_model_parallel_world_size()
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.relative_attention_bias = torch.nn.Embedding(self.relative_attention_num_buckets,
                                                          self.num_attention_heads_per_partition)
        self.is_decoder = is_decoder

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        # shape (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, position_bias=None, old_impl=standard_attention,
                     cross_attention=False, **kw_args):
        log_attention_weights = None
        if not cross_attention:
            if position_bias is None:
                seq_length = q.size(2)
                key_length = k.size(2)
                position_bias = self.compute_bias(key_length, key_length)
                position_bias = position_bias[:, :, -seq_length:, :]
            kw_args['output_cross_layer']['position_bias'] = position_bias
            log_attention_weights = position_bias
        return old_impl(q, k, v, mask, dropout_fn, cross_attention=cross_attention, position_bias=position_bias,
                        log_attention_weights=log_attention_weights, scaling_attention_score=False, **kw_args)


class T5DecoderFinalMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size, tie_word_embeddings=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.tie_word_embeddings = tie_word_embeddings
        if not tie_word_embeddings:
            self.lm_head = VocabParallelEmbedding(
                vocab_size, hidden_size, init_method=unscaled_init_method(0.02))

    def final_forward(self, logits, **kwargs):
        logits_parallel = copy_to_model_parallel_region(logits)
        if self.tie_word_embeddings:
            logits_parallel = logits_parallel * (self.hidden_size ** -0.5)
            logits_parallel = F.linear(logits_parallel, self.transformer.word_embeddings.weight)
        else:
            logits_parallel = F.linear(logits_parallel, self.lm_head.weight)
        return logits_parallel


def t5_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5GatedGeluMLPMixin(BaseMixin):
    def __init__(self, num_layers, hidden_size, inner_hidden_size=None, bias=True, init_method_std=0.02):
        super().__init__()
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.init_method_std = init_method_std
        self.gated_h_to_4h_list = torch.nn.ModuleList([
            ColumnParallelLinear(
                self.hidden_size,
                self.inner_hidden_size,
                gather_output=False,
                init_method=self._init_weights,
                bias=bias,
                module=self,
                name="gated_h_to_4h"
            )
            for layer_id in range(num_layers)])

    def _init_weights(self, weight, **kwargs):
        torch.nn.init.normal_(weight, mean=0, std=self.init_method_std * (self.hidden_size ** -0.5))

    def mlp_forward(self, hidden_states, layer_id=None, **kw_args):
        mlp_module = self.transformer.layers[layer_id].mlp
        hidden_gelu = t5_gelu(mlp_module.dense_h_to_4h(hidden_states))
        hidden_linear = self.gated_h_to_4h_list[layer_id](hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        output = mlp_module.dense_4h_to_h(hidden_states)

        if self.training:
            output = mlp_module.dropout(output)
        return output


class T5Model(EncoderDecoderModel):
    def __init__(self, args, **kwargs):
        self.init_method_std = args.init_method_std
        super().__init__(args, tie_word_embeddings=True, **kwargs, use_bias=False,
                         layernorm=T5LayerNorm, activation_func=torch.nn.functional.relu,
                         init_method=self._init_weights)
        self.encoder.add_mixin(
            "t5-attention", T5AttentionMixin(args.relative_attention_num_buckets, args.num_attention_heads)
        )
        self.encoder.add_mixin(
            "t5-position", T5PositionEmbeddingMixin()
        )
        del self.encoder.transformer.position_embeddings
        num_attention_heads = args.dec_num_attention_heads if args.dec_num_attention_heads is not None else args.num_attention_heads
        self.decoder.add_mixin(
            "t5-attention", T5AttentionMixin(args.relative_attention_num_buckets, num_attention_heads, is_decoder=True)
        )
        self.decoder.add_mixin(
            "t5-position", T5PositionEmbeddingMixin()
        )
        self.decoder.add_mixin(
            "t5-final",
            T5DecoderFinalMixin(args.vocab_size, args.hidden_size, tie_word_embeddings=not args.no_share_embeddings)
        )
        del self.decoder.transformer.position_embeddings
        if args.gated_gelu_mlp:
            self.encoder.add_mixin(
                "gated-mlp", T5GatedGeluMLPMixin(args.num_layers, args.hidden_size, init_method_std=self.init_method_std,
                                                 inner_hidden_size=args.inner_hidden_size, bias=False)
            )
            self.decoder.add_mixin(
                "gated-mlp", T5GatedGeluMLPMixin(args.num_layers, args.hidden_size, init_method_std=self.init_method_std,
                                                 inner_hidden_size=args.inner_hidden_size, bias=False)
            )

    def _init_weights(self, weight, module, name):
        init_method_std = self.init_method_std
        if isinstance(module, MLP):
            if name == "dense_h_to_4h":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.hidden_size ** -0.5))
            elif name == "dense_4h_to_h":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.inner_hidden_size ** -0.5))
            else:
                raise NotImplementedError(name)
        elif isinstance(module, SelfAttention):
            if name == "query_key_value":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.hidden_size ** -0.5))
                torch.nn.init.normal_(weight[:module.inner_hidden_size], mean=0, std=init_method_std * (
                        (module.hidden_size * module.hidden_size_per_attention_head) ** -0.5))
            elif name == "dense":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.inner_hidden_size ** -0.5))
            else:
                raise NotImplementedError(name)
        elif isinstance(module, CrossAttention):
            if name == "query":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (
                        (module.hidden_size * module.hidden_size_per_attention_head) ** -0.5))
            elif name == "key_value":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.hidden_size ** -0.5))
            elif name == "dense":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.inner_hidden_size ** -0.5))
            else:
                raise NotImplementedError(name)
        else:
            raise NotImplementedError(module)

    @classmethod
    def add_model_specific_args(cls, parser):
        super().add_model_specific_args(parser)
        parser.add_argument("--relative-attention-num-buckets", type=int, default=None)
        parser.add_argument("--init-method-std", type=float, default=0.02)
        parser.add_argument("--gated-gelu-mlp", action='store_true')
        parser.add_argument("--no-share-embeddings", action='store_true')

    def encode(self, input_ids, attention_mask=None, **kw_args):
        return super().encode(input_ids, None, attention_mask, **kw_args)

    def decode(self, input_ids, attention_mask=None, encoder_outputs=None, cross_attention_mask=None, **kw_args):
        return super().decode(input_ids, None, attention_mask, encoder_outputs=encoder_outputs,
                              cross_attention_mask=cross_attention_mask, **kw_args)

    def forward(self, enc_input_ids, dec_input_ids, *, enc_attention_mask=None, dec_attention_mask=None,
                cross_attention_mask=None, **kw_args):
        batch_size, seq_length = enc_input_ids.size()[:2]
        if enc_attention_mask is None:
            enc_attention_mask = torch.ones(1, 1, 1, seq_length,
                                            dtype=self.encoder.transformer.word_embeddings.weight.dtype,
                                            device=enc_input_ids.device)
        if cross_attention_mask is None:
            cross_attention_mask = enc_attention_mask
        encoder_outputs = self.encode(enc_input_ids, enc_attention_mask, **kw_args)
        decoder_outputs, *mems = self.decode(dec_input_ids, dec_attention_mask,
                                             encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask,
                                             **kw_args)
        return (encoder_outputs, decoder_outputs, *mems)
