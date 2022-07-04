import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from bmkg.data import MaskTripleDataLoader, RandomChoiceMaskSampler, RandomCorruptMaskSampler
# from ..model import BMKGModel
from model_center.layer import Embedding, Linear, LayerNorm, Encoder
import bmtrain as bmt
from model_center.model import Bert, BertConfig

class DenseGatedACT(bmt.DistributedModule):

    def __init__(self,
                 dim_in: int,
                 dim_ff: int,
                 activate_fn: str = "gelu",
                 dtype=torch.half,
                 int8=False,
                 init_mean=0.0,
                 init_std=0.02,
                 bias=False,
                 length_scale: bool = False,
                 ):
        super().__init__()

        self.w_0 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            length_scale=length_scale,
            length_scale_before=False,
            dtype=dtype,
            int8=int8,
            init_mean=init_mean,
            init_std=init_std,
            bias=bias,
        )

        self.w_1 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            length_scale=length_scale,
            length_scale_before=False,
            dtype=dtype,
            int8=int8,
            init_mean=init_mean,
            init_std=init_std,
            bias=bias,
        )

        if activate_fn == "relu":
            self.act = torch.nn.ReLU()
        elif activate_fn == "gelu":
            self.act = torch.nn.GELU()
        else:
            raise ValueError("Unsupported activation function: %s" % (activate_fn))

    def forward(self, x: torch.Tensor):
        """ This model inherits from bmt.DistributedModule.
            Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.
        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``)
        """
        gelu_score = self.act(self.w_0(x))
        x = self.w_1(x)

        x = gelu_score * x
        return x


class DenseACT(bmt.DistributedModule):

    def __init__(self,
                 dim_in: int,
                 dim_ff: int,
                 activate_fn: str = "gelu",
                 dtype=torch.half,
                 int8=False,
                 init_mean=0.0,
                 init_std=0.02,
                 bias=False,
                 length_scale: bool = False,
                 ):
        super().__init__()

        self.w = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            length_scale=length_scale,
            length_scale_before=False,
            dtype=dtype,
            int8=int8,
            init_mean=init_mean,
            init_std=init_std,
            bias=bias,
        )

        if activate_fn == "relu":
            self.act = torch.nn.ReLU()
        elif activate_fn == "gelu":
            self.act = torch.nn.GELU()
        else:
            raise ValueError("Unsupported activation function: %s" % (activate_fn))

    def forward(self, x: torch.Tensor):
        """ This model inherits from bmt.DistributedModule.
            Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.
        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``)
        """
        x = self.w(x)
        x = self.act(x)

        return x


class FeedForward(bmt.DistributedModule):
    r"""FeedForward module
    Args:
        dim_in (int): input dimension.
        dim_ff (int): middle dimension.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in feed-forward module. Defaults to False.
        activate_fn (str, optional): Defaults to `gated_gelu`.
        dropout_p (int, optional): Defaults to 0.
    """

    def __init__(self,
                 dim_in: int,
                 dim_ff: int,
                 dim_out: int = None,
                 dtype=torch.half,
                 int8=False,
                 init_mean=0.0,
                 init_std=0.02,
                 bias=False,
                 activate_fn="gated_gelu",
                 length_scale: bool = False,
                 dropout_p=0,
                 ):

        super().__init__()

        if activate_fn.startswith("gated_"):
            self.w_in = DenseGatedACT(
                dim_in=dim_in,
                dim_ff=dim_ff,
                activate_fn=activate_fn[6:],
                dtype=dtype,
                int8=int8,
                init_mean=init_mean,
                init_std=init_std,
                bias=bias,
                length_scale=length_scale,
            )
        else:
            self.w_in = DenseACT(
                dim_in=dim_in,
                dim_ff=dim_ff,
                activate_fn=activate_fn,
                dtype=dtype,
                int8=int8,
                init_mean=init_mean,
                init_std=init_std,
                bias=bias,
                length_scale=length_scale,
            )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.layer_norm = LayerNorm(dim_norm=dim_ff, eps=1e-12)

        if dim_out is None:
            dim_out = dim_in

        self.dim_ff = dim_ff
        self.dim_out = dim_out

        self.w_out = Linear(
            dim_in=dim_ff,
            dim_out=dim_out,
            length_scale=length_scale,
            length_scale_before=True,
            dtype=dtype,
            int8=int8,
            init_mean=init_mean,
            init_std=init_std,
            bias=bias,
        )

        self.int8 = int8
        self.length_scale = length_scale

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.
        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """
        x = self.w_in(x)

        if self.dropout is not None:
            x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.w_out(x)

        return x


class CoKE_BMT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # config = BertConfig.from_pretrained("/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/")
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        # self.bert = Bert.from_pretrained("/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/")
        self.bert = Bert.from_pretrained("bert-base-uncased")
        self.dense = Linear(768, 2)
        bmt.init_parameters(self.dense)
        self.mask_id = config['mask_id']
        self.e_mask_id = config['e_mask_id']
        self.mlm_ffn = FeedForward(
            dim_in=768,
            dim_ff=768,
            dim_out=30522,
            init_mean=0.0,
            init_std=0.02,
            dropout_p=0.15
        )

        self.mem_ffn = FeedForward(
            dim_in=768,
            dim_ff=768,
            dim_out=30522,
            init_mean=0.0,
            init_std=0.02,
            dropout_p=0.15
        )

    def forward(self, input_map):
        src_ids = input_map['src_ids'].squeeze()
        input_mask = input_map['input_mask'].squeeze()
        position_ids = input_map['position_ids'].squeeze()
        segment_ids = input_map['segment_ids'].squeeze()
        mlm_mask_pos = input_map['mlm_mask_pos'].squeeze()
        mem_mask_pos = input_map['mem_mask_pos'].squeeze()
        last_hidden_state = self.bert(
            input_ids=src_ids,
            position_ids=position_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
        ).last_hidden_state
        mlm_last_hidden_state = last_hidden_state.view(-1, 768)[mlm_mask_pos.view(-1)]
        mlm_last_hidden_state = self.mlm_ffn(mlm_last_hidden_state.view(-1, 768))
        mem_last_hidden_state = last_hidden_state.view(-1, 768)[mem_mask_pos.view(-1)]
        mem_last_hidden_state = self.mem_ffn(mem_last_hidden_state.view(-1, 768))
        output_map = {
            'mlm_last_hidden_state': mlm_last_hidden_state,
            'mem_last_hidden_state': mem_last_hidden_state
            # 'pooled_output': pooled_output
        }
        return output_map

# import bmtrain as bmt
# bmt.init_distributed(seed=0)
# config = BertConfig.from_pretrained("bert-base-uncased")
# print(config)
# model = BertModel(config)
