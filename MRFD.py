import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        if isinstance(kSize, tuple):
            kSize = kSize[0]
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        output = self.conv(input)
        return output

class CBS(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class SPDConv(nn.Module):
    def __init__(self, inp, outp, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

class MRFD(nn.Module):           # SPDConv + 3*3Conv + AvgMaxConv(论文一)
    def __init__(self, c1, c2):
        super().__init__()
        # 3059708 parameters, 3059692 gradients, 8.3 GFLOPs
        self.spd = SPDConv(c1, 4 * c1)
        self.cbs = CBS(4 * c1, c1)

        self.cbs1 = CBS(c1, c1, 1, 1)

        self.cbs3 = CBS(c1, c1, 3, 2, 1)

        self.cbs2 = CBS(3 * c1, c2, 1, 1)

    def forward(self, x):

        x1 = self.spd(x)
        x1 = self.cbs(x1)

        x2 = self.cbs3(x)

        x3 = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x3 = torch.nn.functional.max_pool2d(x3, 3, 2, 1)
        x3 = self.cbs1(x3)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.cbs2(x)

        return x