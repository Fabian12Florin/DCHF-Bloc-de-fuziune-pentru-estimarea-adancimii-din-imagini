'''
from models.decoders.submodule import *
from models.decoders.submoduleACV import *

class CA_Block_V2_ACV(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None, num_groups=8):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        self.train_image_size = train_image_size
        self.maxdisp = len(norm_disp_range)
        self.num_groups = num_groups

        self.match_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.ELU(),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False)
        )
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.hourglass_att = hourglass_att(8)
        self.corr_feature_att_4 = channelAtt(8, in_channels)
        # Add projection layers
        self.left_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.cost_proj = nn.Conv2d(self.maxdisp, self.maxdisp, 1)
        self.redu = nn.Sequential(
            Conv3x3(in_channels + self.maxdisp, in_channels),
            SELayer(in_channels),
            nn.ELU()
        )

    def forward(self, feats, directs, image_shape):
        left = feats[0]
        right = feats[1]
        B, C, H, W = left.shape

        match_left = self.match_conv(left)
        match_right = self.match_conv(right)

        corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp)  # [B, 1, D, H, W]
        corr_volume = self.corr_stem(corr_volume)  # [B, 8, D, H, W]
        cost_att = self.corr_feature_att_4(corr_volume, left)  # [B, 8, D, H, W]
        imgs = [left, F.avg_pool2d(left, 2), F.avg_pool2d(left, 4), F.avg_pool2d(left, 8)]
        att_weights = self.hourglass_att(cost_att, imgs)  # [B, 1, D, H, W]
        att_weights = F.softmax(att_weights, dim=2)  # softmax over D

        cost = att_weights.squeeze(1)  # [B, D, H, W]

        # Project left and cost to expected channels
        left_proj = self.left_proj(left) if left.shape[1] == self.left_proj.in_channels else \
            nn.Conv2d(left.shape[1], self.left_proj.out_channels, 1, device=left.device, dtype=left.dtype)(left)
        cost_proj = self.cost_proj(cost) if cost.shape[1] == self.cost_proj.in_channels else \
            nn.Conv2d(cost.shape[1], self.cost_proj.out_channels, 1, device=cost.device, dtype=cost.dtype)(cost)

        x = self.redu(torch.cat([left_proj, cost_proj], dim=1))
        return x, cost
'''