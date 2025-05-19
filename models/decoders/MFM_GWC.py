'''
class DAV_Block_Vectorized(nn.Module):
    def __init__(self, in_channels, patch_size=3):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2)

    def forward(self, q, warped_feat):
        B, C, H, W = q.shape
        D = warped_feat.shape[-1]

        q_patches = self.unfold(q)  # [B, C*P*P, H*W]
        q_patches = F.normalize(q_patches, dim=1).permute(0, 2, 1)  # [B, H*W, F]

        k_patches = []
        for d in range(D):
            k = warped_feat[..., d]
            k_patch = self.unfold(k)
            k_patch = F.normalize(k_patch, dim=1).permute(0, 2, 1).unsqueeze(-1)  # [B, H*W, F, 1]
            k_patches.append(k_patch)
        k_patches = torch.cat(k_patches, dim=-1)  # [B, H*W, F, D]

        sim = torch.einsum('bnf,bnfd->bnd', q_patches, k_patches)  # [B, H*W, D]
        sim = sim.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        return sim

def build_gwc_volume(left, right, max_disp, num_groups):
    B, C, H, W = left.shape
    # Ajustează numărul de grupuri dacă nu e divizor al lui C
    if C % num_groups != 0:
        valid_groups = [g for g in range(1, C+1) if C % g == 0]
        num_groups = max([g for g in valid_groups if g <= num_groups])
        print(f"[GWC] Adjusted num_groups to {num_groups} for C={C}")

    group_channels = C // num_groups
    volume = left.new_zeros(B, num_groups, max_disp, H, W)

    for d in range(max_disp):
        if d > 0:
            l = left[:, :, :, d:]
            r = right[:, :, :, :-d]
            corr = (l * r).view(B, num_groups, group_channels, H, W - d).mean(2)
            volume[:, :, d, :, d:] = corr
        else:
            corr = (left * right).view(B, num_groups, group_channels, H, W).mean(2)
            volume[:, :, d, :, :] = corr

    return volume

class MFM_DAV_GWC_Fusion(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None, patch_size=3, num_groups=32):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        self.train_image_size = train_image_size
        self.max_disp = len(norm_disp_range)
        self.patch_size = patch_size
        self.num_groups = num_groups

        self.q_conv = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.k_conv = nn.Conv2d(in_channels, in_channels // 4, 1)

        self.dav = DAV_Block_Vectorized(in_channels // 4, patch_size=patch_size)
        self.gwc_patch = nn.Conv3d(num_groups, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False)

        self.norm_t = nn.GroupNorm(1, in_channels)
        self.norm_dav = nn.GroupNorm(1, self.max_disp)
        self.norm_gwc = nn.GroupNorm(1, self.max_disp)

        self.expand_dav = nn.Conv2d(self.max_disp, in_channels, 1)
        self.expand_gwc = nn.Conv2d(self.max_disp, in_channels, 1)

        self.alpha_dav = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1), nn.Sigmoid())
        self.alpha_gwc = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1), nn.Sigmoid())

        self.redu = MiniNAFBlock(in_channels)

    def forward(self, feats, directs, image_shape):
        t_feat, s_feat = feats
        q = self.q_conv(t_feat)
        k = self.k_conv(s_feat)

        warped_k = self._get_warped_frame(k, directs, image_shape, q.shape[2:])

        dav_cost = self.dav(q, warped_k)  # [B, D, H, W]
        dav_cost = self.norm_dav(dav_cost)
        dav_feat = self.expand_dav(dav_cost)

        gwc_volume = build_gwc_volume(q, k, self.max_disp, self.num_groups)  # [B, G, D, H, W]
        gwc_volume = self.gwc_patch(gwc_volume).squeeze(1)  # [B, D, H, W]
        gwc_volume = self.norm_gwc(gwc_volume)
        gwc_feat = self.expand_gwc(gwc_volume)

        t_norm = self.norm_t(t_feat)
        alpha_dav = self.alpha_dav(torch.cat([t_norm, dav_feat], dim=1))
        alpha_gwc = self.alpha_gwc(torch.cat([t_norm, gwc_feat], dim=1))

        fused = (1 - alpha_dav - alpha_gwc) * t_norm + alpha_dav * dav_feat + alpha_gwc * gwc_feat
        out = self.redu(fused)
        return out, dav_cost

    def _get_warped_frame(self, x, directs, image_shape, target_hw):
        B, C, Hx, Wx = x.shape
        Ht, Wt = target_hw
        D = len(self.norm_disp_range)

        base_coord = F.affine_grid(torch.eye(2, 3).unsqueeze(0).to(x.device), [1, 1, Hx, Wx], align_corners=True)
        zeros = torch.zeros_like(base_coord)

        rel_scale = self.train_image_size[1] / image_shape[3] if self.train_image_size else 1
        frames = []
        for disp in self.norm_disp_range:
            disp_map = zeros.clone()
            disp_map[..., 0] += disp * 2 * rel_scale
            grid = disp_map * directs + base_coord
            warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
            frames.append(warped.unsqueeze(2))
        warped_feat = torch.cat(frames, dim=2)  # [B, C, D, H, W]
        warped_feat = warped_feat.permute(0, 1, 3, 4, 2)  # [B, C, H, W, D]
        return warped_feat
'''