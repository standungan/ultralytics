# ultralytics/nn/modules/ms_head.py
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect

# Try to import the repo CBAM / myCBAM first, otherwise fall back to tiny local CBAM
try:
    from ultralytics.nn.modules.conv import myCBAM as _RepoCBAM
except Exception:
    try:
        from ultralytics.nn.modules.conv import CBAM as _RepoCBAM
    except Exception:
        _RepoCBAM = None

class _TinyCBAM(nn.Module):
    """Tiny lightweight CBAM fallback if repo doesn't expose CBAM."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 1), channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.ca(x) * x
        max_pool, _ = torch.max(ca, dim=1, keepdim=True)
        avg_pool = torch.mean(ca, dim=1, keepdim=True)
        sa = self.sa(torch.cat([max_pool, avg_pool], dim=1))
        return ca * sa

class DilatedRefine(nn.Module):
    """Small dilated conv refinement for finest scale."""
    def __init__(self, in_ch: int, out_ch: int = None, kernel=3, dilation=2, act=nn.SiLU):
        super().__init__()
        out_ch = out_ch or in_ch
        pad = dilation * (kernel // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MultiScaleDetect(nn.Module):
    """
    Wrapper that refines multi-scale features (P2,P3,P4,P5) and calls Ultralytics Detect.
    Args:
        nc: number of classes
        ch: tuple/list of channels for each feature map in order (P2,P3,P4,P5)
        use_cbam_for_all: whether to apply CBAM on P3,P4,P5 (True/False)
        refine_p2: whether to apply CBAM + dilated refine on P2
    """
    def __init__(self, nc: int = 3, ch: tuple = (), use_cbam_for_all: bool = True, refine_p2: bool = True):
        super().__init__()
        if isinstance(ch, (list, tuple)):
            self.ch = tuple(int(x) for x in ch)
        else:
            raise ValueError("ch must be a tuple/list of 4 channel sizes (P2,P3,P4,P5)")
        assert len(self.ch) == 4, "MultiScaleDetect expects 4 channel entries (P2,P3,P4,P5)"

        self.nc = int(nc)
        # underlying Detect head (keeps loss/assignment intact)
        self.detect = Detect(self.nc, ch=self.ch)

        # cbam impl
        cbam_cls = _RepoCBAM if _RepoCBAM is not None else _TinyCBAM

        # Build refinement modules for each head
        self.refine_modules = nn.ModuleList()
        for i, c in enumerate(self.ch):
            if i == 0 and refine_p2:
                # P2: CBAM -> DilatedRefine (strong refinement)
                self.refine_modules.append(nn.Sequential(
                    cbam_cls(c),
                    DilatedRefine(c, c)
                ))
            else:
                # P3/P4/P5: optional CBAM or identity
                if use_cbam_for_all:
                    self.refine_modules.append(cbam_cls(c))
                else:
                    self.refine_modules.append(nn.Identity())

    def forward(self, x: list):
        """
        x: list/tuple of 4 tensors [P2, P3, P4, P5] in that order.
        Returns: whatever Detect.forward returns (training: list of raw preds; inference: decoded detections)
        """
        if not isinstance(x, (list, tuple)):
            raise ValueError("MultiScaleDetect expects list/tuple of feature maps [P2,P3,P4,P5]")
        if len(x) != len(self.refine_modules):
            raise ValueError(f"Expected {len(self.refine_modules)} feature maps, got {len(x)}")

        xr = []
        for i, xi in enumerate(x):
            xr.append(self.refine_modules[i](xi))
        # delegate to Detect for prediction/loss handling
        return self.detect(xr)
