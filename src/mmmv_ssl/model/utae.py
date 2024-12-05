# pylint: skip-file
# pylint: disable=invalid-name
# Not our file, but was reformatted a bit according to pylint comments
"""
U-TAE Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
from abc import abstractmethod

import torch
from einops import rearrange
from torch import nn

from mmmv_ssl.model.dataclass import OutUTAEForward
from mmmv_ssl.model.ltae2d import LTAE2d


class UTAE(nn.Module):
    # pylint: disable=R0902

    """
    U-TAE architecture for spatio-temporal encoding of satellite image time series.
    Args:
        input_dim (int): Number of channels in the input images.
        encoder_widths (List[int]): List giving the number of channels
        of the successive encoder_widths of the convolutional encoder.
        This argument also defines the number of encoder_widths
        (i.e. the number of downsampling steps +1) in the architecture.
        The number of channels are given from top to bottom,
        i.e. from the highest to the lowest resolution.
        decoder_widths (List[int], optional): Same as encoder_widths but for the decoder.
        The order in which the number of
        channels should be given is also from top to bottom.
        If this argument is not specified the decoder
        will have the same configuration as the encoder.
        out_conv (List[int]): Number of channels of the successive convolutions for the
        str_conv_k (int): Kernel size of the strided up and down convolutions.
        str_conv_s (int): Stride of the strided up and down convolutions.
        str_conv_p (int): Padding of the strided up and down convolutions.
        agg_mode (str): Aggregation mode for the skip connections. Can either be:
            - att_group (default) : Attention weighted temporal average, using the same
            channel grouping strategy as in the LTAE. The attention masks are bilinearly
            resampled to the resolution of the skipped feature maps.
            - att_mean : Attention weighted temporal average,
             using the average attention scores across heads for each date.
            - mean : Temporal average excluding padded dates.
        encoder_norm (str): Type of normalisation layer to use in the encoding branch.
         Can either be:
            - group : GroupNorm (default)
            - batch : BatchNorm
            - instance : InstanceNorm
        n_head (int): Number of heads in LTAE.
        d_model (int): Parameter of LTAE
        d_k (int): Key-Query space dimension
        encoder (bool): If true, the feature maps instead of the class scores are returned
                        (default False)
        return_maps (bool): If true, the feature maps instead of the class scores are returned
                        (default False)
        pad_value (float): Value used by the dataloader for temporal padding.
        padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
    """

    # pylint: disable=R0913
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            encoder_widths: list[int] | None = None,
            decoder_widths: list[int] | None = None,
            out_conv: list[int] | None = None,
            str_conv_k: int = 4,
            str_conv_s: int = 2,
            str_conv_p: int = 1,
            agg_mode: str = "att_group",
            encoder_norm: str = "group",
            n_head: int = 16,
            d_model: int = 256,
            d_k: int = 4,
            encoder: bool = False,
            return_maps: bool = False,
            pad_value: float = 0,
            padding_mode: str = "reflect",
            # len_max_seq=None,  # unused
    ):

        super().__init__()
        if out_conv is None:
            out_conv = [32]
        if decoder_widths is None:
            decoder_widths = [32, 32, 64, 128]
        if encoder_widths is None:
            encoder_widths = [64, 64, 64, 128]
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0]
            if decoder_widths is not None
            else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths)
            if decoder_widths is not None
            else sum(encoder_widths)
        )
        self.pad_value = pad_value
        self.encoder = encoder
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[in_channels] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = TemporalAggregator(mode=agg_mode)
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0]] + out_conv + [out_channels],
            padding_mode=padding_mode,
        )

    def forward(
            self,
            batch: torch.Tensor,
            batch_positions: torch.Tensor = None,
            key_padding_mask: torch.Tensor = None,
            return_attns: bool = False,
    ) -> OutUTAEForward:
        if key_padding_mask is None:
            pad_mask = (
                (batch == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
            )  # BxT pad mask
        else:
            pad_mask = key_padding_mask

            assert (
                    len(pad_mask.shape) == 2
            ), "Wrong encoding of padd mask should be (B*T) not {}".format(
                pad_mask.shape
            )
        out = self.in_conv.smart_forward(batch)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(
            feature_maps[-1],
            batch_positions=batch_positions,
            pad_mask=pad_mask,
        )
        # SPATIAL DECODER
        maps = None
        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            out = rearrange(out, "b c h w -> b h w c")
            return OutUTAEForward(out, attn=None, feature_maps=maps)

        out = self.out_conv(out)
        out = rearrange(out, "b c h w -> b h w c")
        if return_attns:
            return OutUTAEForward(out, attn=att, feature_maps=None)
        if self.return_maps:
            return OutUTAEForward(out, feature_maps=maps)
        return OutUTAEForward(out)


class TemporalAggregator(nn.Module):
    def __init__(self, mode: str = "mean"):
        """Temporal aggregator class"""
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor,
                pad_mask: torch.Tensor | None = None,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass."""
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape  # pylint: disable=C0103
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = (
                        out.sum(dim=1)
                        / (~pad_mask).sum(dim=1)[:, None, None, None]
                )
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape  # pylint: disable=C0103
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW

                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value: float | None = None):
        super().__init__()
        self.out_shape = None
        self.pad_value = pad_value

    @abstractmethod
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Generic forward pass of the model."""

    def smart_forward(self, batch: torch.Tensor) -> torch.Tensor:
        if len(batch.shape) == 4:
            return self.forward(batch)

        b, t, c, h, w = batch.shape  # pylint: disable=C0103

        if self.pad_value is not None:
            dummy = torch.zeros(batch.shape, device=batch.device).float()
            self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

        out = batch.view(b * t, c, h, w)
        if self.pad_value is not None:
            pad_mask = (
                (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
            )
            if pad_mask.any():
                temp = (
                        torch.ones(
                            self.out_shape,
                            device=batch.device,
                            requires_grad=False,
                        )
                        * self.pad_value
                )
                temp[~pad_mask] = self.forward(out[~pad_mask])
                out = temp
            else:
                out = self.forward(out)
        else:
            out = self.forward(out)
        _, c, h, w = out.shape  # pylint: disable=C0103
        out = out.view(b, t, c, h, w)
        return out


class ConvLayer(nn.Module):
    def __init__(
            self,
            nkernels: list[int],
            norm: str = "batch",
            k: int = 3,
            s: int = 1,
            p: int = 1,
            n_groups: int = 4,
            last_relu: bool = True,
            padding_mode: str = "reflect",
    ):
        super().__init__()
        layers = []
        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d
        elif norm == "group":

            def group_norm(num_feats: int) -> nn.GroupNorm:
                return nn.GroupNorm(
                    num_channels=num_feats,
                    num_groups=n_groups,
                )

            norm_layer = group_norm
        else:
            norm_layer = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if norm_layer is not None:
                layers.append(norm_layer(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.conv(batch)


class ConvBlock(TemporallySharedBlock):
    def __init__(
            self,
            nkernels: list[int],
            pad_value: float | None = None,
            norm: str = "batch",
            last_relu: bool = True,
            padding_mode: str = "reflect",
    ):
        """Convolutional Block of Unet"""

        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.conv(batch)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            k: int,
            s: int,
            p: int,
            pad_value: float | None = None,
            norm: str = "batch",
            padding_mode: str = "reflect",
    ):
        """Down Convolutional Block of Unet"""
        super().__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        out = self.down(batch)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            k: int,
            s: int,
            p: int,
            norm: str = "batch",
            d_skip: int | None = None,
            padding_mode: str = "reflect",
    ):
        """Up Convolutional Block of Unet"""

        super().__init__()
        dims = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=1),
            nn.BatchNorm2d(dims),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in,
                out_channels=d_out,
                kernel_size=k,
                stride=s,
                padding=p,
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + dims, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, batch: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        out = self.up(batch)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out
