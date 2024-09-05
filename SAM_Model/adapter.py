import math

import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout=0.0,
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.input_dim)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.input_dim, self.output_dim)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.output_dim, self.input_dim)

        self.dropout = dropout

    def forward(self, x):
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        output = up * self.scale

        if self.adapter_layernorm_option == 'out':
            output = self.adapter_layer_norm_before(output)

        return output


class LoRALinear(nn.Module):
    r"""Implements LoRA in a linear layer.

    Args:
        original_layer (nn.Linear): The linear layer to be finetuned.
        alpha (int): The scale factor of LoRA. Defaults to 1.
        rank (int): The rank of LoRA. Defaults to 0.
        drop_rate (float): The drop out rate for LoRA. Defaults to 0.

    Note:
        The forward process of LoRA linear layer is:

        .. math::
            `y = W_0 x + BAx * (\alpha / r)`

        Where :math:`x` is the input, :math:`y` is the output,
        :math:`W_0` is the parameter of the original layer,
        :math:`A` and :math:`B` are the low-rank decomposition matrixs,
        :math: `\alpha` is the scale factor and :math: `r` is the rank.
    """

    def __init__(self,
                 original_layer: nn.Linear,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.):
        super(LoRALinear, self).__init__()
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.original_layer = original_layer

    def forward(self, x: torch.Tensor):
        out = self.original_layer(x)

        lora_x = self.lora_dropout(x)
        lora_out = self.lora_up(self.lora_down(lora_x)) * self.scaling

        return out + lora_out


class LoRA(nn.Module):
    r"""Implements LoRA in a linear layer.

    Args:
        original_layer (nn.Linear): The linear layer to be finetuned.
        alpha (int): The scale factor of LoRA. Defaults to 1.
        rank (int): The rank of LoRA. Defaults to 0.
        drop_rate (float): The drop out rate for LoRA. Defaults to 0.

    Note:
        The forward process of LoRA linear layer is:

        .. math::
            `y = W_0 x + BAx * (\alpha / r)`

        Where :math:`x` is the input, :math:`y` is the output,
        :math:`W_0` is the parameter of the original layer,
        :math:`A` and :math:`B` are the low-rank decomposition matrixs,
        :math: `\alpha` is the scale factor and :math: `r` is the rank.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.):
        super().__init__()
        in_features = input_dim
        out_features = output_dim

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # self.original_layer = original_layer

    def forward(self, x: torch.Tensor):
        # out = self.original_layer(x)

        lora_x = self.lora_dropout(x)
        lora_out = self.lora_up(self.lora_down(lora_x)) * self.scaling

        # return out + lora_out
        return lora_out
