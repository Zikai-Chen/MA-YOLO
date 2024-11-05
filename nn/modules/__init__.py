# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""
from .MLCA import *
from .iRMB import *
from .MSBlock import *
from .CSPHet import *
from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, C2f_ScConv, DySnakeConvC2f, C2f_FLA, C2f_ODConv, C2f_PConv, C2f_ECAPConv, C2f_ECAPConvQZ)
                   #C2f_SCCon、DySnakeConvC2f、C2f_FLA, C2f_ODConv、C2f_PConv、C2f_ECAPConv、C2f_ECAPConvQZ
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, BiLevelRoutingAttention, SPDConv, SPDConvn)    #Biformer、SPDConv
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP',
           'BiLevelRoutingAttention', 'C2f_ScConv', 'SPDConv', 'DySnakeConvC2f', 'C2f_FLA', 'C2f_ODConv', 'C2f_PConv', 'C2f_ECAPConv', 'C2f_ECAPConvQZ', 'SPDConvn',
           'CSPHet')
            #Biformer、C2f_SCConv、SPDConv、DySnakeConvC2f、C2f_FLA, C2f_ODConv、C2f_PConv、C2f_ECAPConv、C2f_ECAPConvQZ
