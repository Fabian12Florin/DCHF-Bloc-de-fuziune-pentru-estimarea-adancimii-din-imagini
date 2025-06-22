import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")

class UncertaintyDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec, scales=range(3), num_output_channels=1, use_skips=True):
        super(UncertaintyDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_scales = len(scales)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array(num_ch_dec)

        # decoder
        self.uncert = OrderedDict()
        self.convs = OrderedDict()
        for i in range(self.num_scales, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.num_scales else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.uncert[("unvertaconv",s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()) + list(self.uncert.values()))
        
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
                    
        for i in range(self.num_scales, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                skip = input_features[i - 1]
                if x[0].shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size = x[0].shape[2:], mode='nearest')
                x.append(skip)
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                uncerts = self.uncert[("unvertaconv",i)](x)
                #self.outputs[("uncert",i)] = F.elu_(uncerts, alpha=1.0) + 1.0 + 1e-3
                self.outputs[("uncert",i)] = self.sigmoid(uncerts)
                #self.outputs[("uncert",i)] = self.softplus(uncerts) + 1e-6
                
        return self.outputs 