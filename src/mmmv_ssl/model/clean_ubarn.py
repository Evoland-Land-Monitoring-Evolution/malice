import logging
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from mt_ssl.constant.dataset import ENCODE_PERMUTATED
from mt_ssl.data.bert_batch import InputBERTBatch
from mt_ssl.data.mask import permutate_flagged_patches
from mt_ssl.data.mt_batch import BInput5d, BOutputUBarn
from mt_ssl.model.encoding import InputEncoding, LearnedPE
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

my_logger = logging.getLogger(__name__)


class CleanUBarn(nn.Module):
    def __init__(
        self,
        ne_layers: int,
        d_model: int = 256,
        d_hidden: int = 512,
        dropout: float = 0.1,
        block_name: Literal["se", "basicconv", "pff", "last"] = "pff",
        norm_first: bool = False,
        input_channels: int = 10,
        nhead: int = 4,
        attn_dropout: float = 0.1,
        encoding_config: DictConfig | None = None,
        max_len_pe: int = 3000,
        pe_cst: int = 10000,
        pe_module: nn.Module | None = None,
        use_transformer: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ne_layers = ne_layers
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.block_name = block_name
        self.norm_first = norm_first
        self.input_channels = input_channels
        self.nhead = nhead
        self.attn_dropout = attn_dropout
        self.encoding_config = encoding_config
        self.max_len_pe = max_len_pe
        self.pe_cst = pe_cst
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hidden,
            dropout=dropout,
            norm_first=norm_first,
            batch_first=True,
        )
        if use_transformer:
            self.temporal_encoder = TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=ne_layers
            )
        else:
            self.temporal_encoder = None
        print(f"PE module {pe_module} type {type(pe_module)}")
        self.patch_encoding = InputEncoding(
            inplanes=input_channels,
            planes=d_model,
            dropout=dropout,
            model_config=encoding_config,
            pe_cst=pe_cst,
            pe_module=pe_module,
        )

    def forward(
        self,
        batch_input: BInput5d | InputBERTBatch,
        return_attns: bool,
        apply_corruption: bool = False,
    ) -> BOutputUBarn:
        """

        Args:
            batch_input ():
            return_attns ():

        Returns:

        """
        x, doy_encoding = self.patch_encoding(
            batch_input.sits, batch_input.input_doy, batch_input.true_doy
        )
        if apply_corruption and isinstance(batch_input, InputBERTBatch):
            x = self.masking(
                x,
                mask_encoding=batch_input.corruption_mask,
                pad_index=batch_input.padd_index,
            )
        my_logger.debug(f"x{x.shape} doy {doy_encoding.shape}")
        if self.temporal_encoder is not None:
            x = x + doy_encoding
            print(batch_input.padd_index.shape)
            print(x.shape)
            b, t, c, h, w = x.shape
            padd_index = repeat(
                batch_input.padd_index, "b t -> b t h w ", h=h, w=w
            )
            padd_index = rearrange(padd_index, " b t h w -> (b h w ) t")
            x = rearrange(x, "b t c h w -> (b h w ) t c")
            x = self.temporal_encoder(
                x,
                src_key_padding_mask=padd_index,
            )
            x = rearrange(x, "(b h w ) t c -> b t c h w ", b=b, h=h, w=w)
            return BOutputUBarn(x)

        return BOutputUBarn(x, None)

    def masking(
        self, myinput: Tensor, mask_encoding: Tensor, pad_index: Tensor
    ):
        """

        Args:
            myinput ():
            mask_encoding ():
            pad_index ():

        Returns: apply the [MASK]ing strategy of the BERT: permutates the pixel value for the selected patch
        """
        x = myinput.clone()
        padd_val_dim = torch.count_nonzero(pad_index, dim=1)
        pad_val = torch.max(padd_val_dim)  # TODO is it really useful ??
        if pad_val != 0:
            nopad_x = x[:, :-pad_val, :, :, :]
        else:
            nopad_x = x
        nopad_x_shape = torch.Tensor(list(nopad_x.shape)).to(
            device=x.device, dtype=torch.int32
        )
        b, n, hdim, h, w = nopad_x_shape
        mask_encoding = torch.repeat_interleave(
            mask_encoding, repeats=hdim, dim=2
        )  # b t c h w "

        return permutate_flagged_patches(
            x, (mask_encoding == ENCODE_PERMUTATED)
        )


@dataclass
class ConfigUbarn:
    _target_: str = "mt_ssl.model.ubarn.UBarn"
    ne_layers: int = 3
    d_model: int = 256
    d_hidden: int = (512,)
    dropout: float = (0.1,)
    block_name: Literal["se", "basicconv", "pff", "last"] = ("pff",)
    norm_first: bool = (False,)
    input_channels: int = (10,)
    nhead: int = (4,)
    attn_dropout: float = (0.1,)
    encoding_config: DictConfig | None = (None,)
    pe_module: LearnedPE | None = None
    max_len_pe: int = (3000,)
    pe_cst: int = 10000
    use_transformer: bool = True
    args: list = None
    kwargs: dict = None
