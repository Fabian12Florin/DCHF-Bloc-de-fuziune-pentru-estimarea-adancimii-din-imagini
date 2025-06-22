import torch
import torch.nn as nn
import torch.nn.functional as F

class channelAtt(nn.Module):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()
        mid_chan = max(1, im_chan // 2)
        self.im_att = nn.Sequential(
            BasicConv(im_chan, mid_chan, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(mid_chan, cv_chan, 1)
        )

    def forward(self, cv, im):
        # Ensure im spatial size matches cv
        if im.shape[-2:] != cv.shape[-2:]:
            im = F.interpolate(im, size=cv.shape[-2:], mode='bilinear', align_corners=False)
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att) * cv
        return cv

class hourglass_att(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_att, self).__init__()

        # Only downsample H and W, keep D fixed!
        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=(1,2,2)),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1)
        )
        self.conv2 = nn.Sequential(
            BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=(1,2,2)),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1)
        )
        self.conv3 = nn.Sequential(
            BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=(1,2,2)),
            BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1)
        )

        # Only upsample H and W, keep D fixed!
        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(1,4,4), padding=(0,1,1), stride=(1,2,2))
        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(1,4,4), padding=(0,1,1), stride=(1,2,2))
        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False, relu=False, kernel_size=(1,4,4), padding=(0,1,1), stride=(1,2,2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1)
        )
        self.agg_1 = nn.Sequential(
            BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x, imgs):
        # imgs: list of [B, C, H, W] at different scales
        conv1 = self.conv1(x)
        att_8 = channelAtt(conv1.shape[1], imgs[1].shape[1]).to(conv1.device)
        conv1 = att_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        att_16 = channelAtt(conv2.shape[1], imgs[2].shape[1]).to(conv2.device)
        conv2 = att_16(conv2, imgs[2])

        conv3 = self.conv3(conv2)
        att_32 = channelAtt(conv3.shape[1], imgs[3].shape[1]).to(conv3.device)
        conv3 = att_32(conv3, imgs[3])

        conv3_up = self.conv3_up(conv3)
        if conv3_up.shape[2:] != conv2.shape[2:]:
            conv3_up = F.interpolate(conv3_up, size=conv2.shape[2:], mode='trilinear', align_corners=False)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        att_up_16 = channelAtt(conv2.shape[1], imgs[2].shape[1]).to(conv2.device)
        conv2 = att_up_16(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2)
        if conv2_up.shape[2:] != conv1.shape[2:]:
            conv2_up = F.interpolate(conv2_up, size=conv1.shape[2:], mode='trilinear', align_corners=False)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        att_up_8 = channelAtt(conv1.shape[1], imgs[1].shape[1]).to(conv1.device)
        conv1 = att_up_8(conv1, imgs[1])

        conv = self.conv1_up(conv1)
        return conv
    
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x
    
def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def build_absdiff_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = torch.abs(refimg_fea[:, :, :, i:] - targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = torch.abs(refimg_fea - targetimg_fea)
    volume = volume.mean(1, keepdim=True)  # mean over channels, shape [B, 1, D, H, W]
    return volume.contiguous()