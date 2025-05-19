
'''
class DSSCAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.scale = in_channels ** -0.5

        self.norm_q = LayerNorm2d(in_channels)
        self.norm_k = LayerNorm2d(in_channels)

        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_proj_s = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))

    def forward(self, t_feat, s_feat):
        B, C, H, W = t_feat.shape
        HW = H * W

        # Normalize and project
        q = self.q_proj(self.norm_q(t_feat)).flatten(2).transpose(1, 2)  # [B, HW, C]
        k = self.k_proj(self.norm_k(s_feat)).flatten(2)                  # [B, C, HW]
        v = self.v_proj_s(s_feat).flatten(2).transpose(1, 2)             # [B, HW, C]

        # Attention
        attn = torch.bmm(q, k) * self.scale                              # [B, HW, HW]
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v).transpose(1, 2).contiguous()            # [B, C, HW]
        out = out.view(B, C, H, W)

        updated_t = t_feat + out * self.beta
        return updated_t

class CA_Block_V2_DSSCAM(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        self.train_image_size = train_image_size

        self.fusion = DSSCAM(in_channels)
        self.redu = MiniNAFBlock(in_channels)

    def forward(self, feats, directs, image_shape):
        t_feat, s_feat = feats
        warped = self._get_warped_frame(s_feat, directs, image_shape, t_feat.shape[2:])  # [B, C, H, W, D]

        # ➤ Agregăm peste disparitate (ex: medie)
        warped = warped.mean(dim=-1)  # [B, C, H, W]

        fused = self.fusion(t_feat, warped)
        out = self.redu(fused)
        return out, None


    def _get_warped_frame(self, x, directs, image_shape, target_hw):
        B, C, Hx, Wx = x.shape
        Ht, Wt = target_hw
        D = len(self.norm_disp_range)

        i_theta = torch.zeros(1, 2, 3, device=x.device)
        i_theta[:, 0, 0] = 1
        i_theta[:, 1, 1] = 1
        base_coord = F.affine_grid(i_theta, [1, 1, Hx, Wx], align_corners=True)
        zeros = torch.zeros_like(base_coord)

        rel_scale = self.train_image_size[1] / image_shape[3] if self.train_image_size else 1

        warped_frames = []
        for disp in self.norm_disp_range:
            disp_map = zeros.clone()
            disp_map[..., 0] += disp * 2 * rel_scale
            grid = disp_map * directs + base_coord
            warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
            warped_frames.append(warped.unsqueeze(2))

        warped_feat = torch.cat(warped_frames, dim=2)  # [B, C, H, W, D]

        if (Hx, Wx) != (Ht, Wt):
            warped_feat = warped_feat.permute(0, 4, 1, 2, 3)  # [B, D, C, H, W]
            warped_feat = warped_feat.reshape(B * D, C, Hx, Wx)
            warped_feat = F.interpolate(warped_feat, size=(Ht, Wt), mode='bilinear', align_corners=False)
            warped_feat = warped_feat.view(B, D, C, Ht, Wt).permute(0, 2, 3, 4, 1)

        return warped_feat
'''


'''
# Prima modificare AFB
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        device = x.device

        y_embed = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_embed = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

        pos = torch.cat([x_embed, y_embed], dim=1)
        pos = pos.repeat(1, C // 2, 1, 1)[:, :C, :, :]
        return x + pos

class AFB_Block(nn.Module):
    def __init__(self, in_channels, norm_disp_range=None, train_image_size=None, heads=2):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = in_channels // 2
        self.heads = heads

        self.pe = PositionalEncoding2D(in_channels)

        self.q_proj = nn.Linear(in_channels, self.embed_dim)
        self.k_proj = nn.Linear(in_channels, self.embed_dim)
        self.v_proj = nn.Linear(in_channels, self.embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, in_channels)
        )

    def forward(self, feats, directs=None, image_shape=None):
        t_feat, s_feat = feats[0], feats[1]  # (B, C, H, W)
        B, C, H, W = t_feat.shape

        t_feat = self.pe(t_feat)
        s_feat = self.pe(s_feat)

        t_flat = t_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        s_flat = s_feat.flatten(2).transpose(1, 2)

        Q = self.q_proj(t_flat)
        K = self.k_proj(s_flat)
        V = self.v_proj(s_flat)

        attn_out, _ = self.attn(Q, K, V)  # (B, HW, embed_dim)

        x = self.ffn(attn_out)  # (B, HW, C)
        x = x.transpose(1, 2).view(B, C, H, W)

        return x, None  # keep same API as CA_Block_V2
'''

'''
# A doua modificare - Mamba like (gate+sigmoid) epipolar, pentru tot blocul
class MambaSimple1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):  # x: (B, H, W, C)
        B, H, W, C = x.shape
        x = x.reshape(B * H, W, C)
        x_norm = self.norm(x)
        gate = torch.sigmoid(self.gate_proj(x_norm))
        value = self.value_proj(x_norm)
        gated = gate * value
        out = gated + x
        out = self.ffn(out)
        return out.view(B, H, W, C)

class MFM_MambaSimpleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.fuse_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.mamba = MambaSimple1D(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.cost_conv = nn.Conv2d(in_channels, 49, kernel_size=3, padding=1)

    def forward(self, feats, directs=None, image_shape=None):
        t_feat, s_feat = feats

        fuse_input = torch.cat([t_feat, s_feat], dim=1)
        fused = self.fuse_conv(fuse_input)

        x = fused.permute(0, 2, 3, 1)
        x = self.mamba(x)
        x = x.permute(0, 3, 1, 2)

        x = x + t_feat
        x = self.norm(x)
        fused_feat = self.out_conv(x)

        cost_3_s = self.cost_conv(fused_feat)  # (B, 49, H, W)
        return fused_feat, cost_3_s
'''

'''
# A doua modificare - Mamba like pe doua directii
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y

class MambaSimple1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):  # x: (B, H, W, C)
        B, H, W, C = x.shape
        x = x.reshape(B * H, W, C)
        x_norm = self.norm(x)
        gate = torch.sigmoid(self.gate_proj(x_norm))
        value = self.value_proj(x_norm)
        gated = gate * value
        out = gated + x
        out = self.ffn(out)
        return out.view(B, H, W, C)

class MFM_MambaSimpleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.fuse_conv = nn.Conv2d(in_channels * 2 + 2, in_channels, kernel_size=1)
        self.mamba = MambaSimple1D(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.se = SELayer(in_channels)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.cost_conv = nn.Conv2d(in_channels, 49, kernel_size=3, padding=1)

    def forward(self, feats, directs=None, image_shape=None):
        t_feat, s_feat = feats

        fuse_input = torch.cat([t_feat, s_feat], dim=1)

        if directs is not None:
            hint_disp = directs  # (B, 1, H, W)
            hint_disp = F.interpolate(hint_disp, size=t_feat.shape[2:], mode='bilinear', align_corners=True)
            hint_mask = (hint_disp > 0).float()
            fuse_input = torch.cat([fuse_input, hint_disp, hint_mask], dim=1)


        fused = self.fuse_conv(fuse_input)

        x = fused.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.mamba(x)
        x = x.permute(0, 3, 1, 2)

        x = x + t_feat
        x = self.norm(x)
        x = self.se(x)
        fused_feat = self.out_conv(x)

        cost_3_s = self.cost_conv(fused_feat)
        return fused_feat, cost_3_s
'''

'''
# A treia modificare
class Mamba1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):  # x: (B, L, C)
        x_norm = self.norm(x)
        gate = torch.sigmoid(self.gate_proj(x_norm))
        value = self.value_proj(x_norm)
        gated = gate * value
        out = gated + x
        out = self.ffn(out)
        return out

class MFM_SS2DSimpleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.fuse_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.mamba_h = Mamba1D(in_channels)
        self.mamba_v = Mamba1D(in_channels)

        self.norm = nn.BatchNorm2d(in_channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.cost_conv = nn.Conv2d(in_channels, 49, kernel_size=3, padding=1)

    def forward(self, feats, directs=None, image_shape=None):
        t_feat, s_feat = feats  # (B, C, H, W)

        # 1. Concat și fuse
        fuse_input = torch.cat([t_feat, s_feat], dim=1)  # (B, 2C, H, W)
        fused = self.fuse_conv(fuse_input)               # (B, C, H, W)

        B, C, H, W = fused.shape

        # 2. Scan orizontal (pe W)
        x_h = fused.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_h = x_h.view(B * H, W, C)
        out_h = self.mamba_h(x_h).view(B, H, W, C)

        # 3. Scan vertical (pe H)
        x_v = fused.permute(0, 3, 2, 1).contiguous()  # (B, W, H, C)
        x_v = x_v.view(B * W, H, C)
        out_v = self.mamba_v(x_v).view(B, W, H, C).permute(0, 2, 1, 3)  # back to (B, H, W, C)

        # 4. Combinație direcții
        out = out_h + out_v  # (B, H, W, C)

        # 5. Format CNN
        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
        out = out + t_feat
        out = self.norm(out)
        fused_feat = self.out_conv(out)

        cost_3_s = self.cost_conv(fused_feat)
        return fused_feat, cost_3_s
'''

'''
# Mamba cu ssm, inlocuieste redu
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()

class SS2D_Mamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.param_gen = nn.Conv1d(dim, dim * 4, kernel_size=1)

    def forward_scan(self, x, direction):
        B, C, H, W = x.shape
        if direction == 'horizontal':
            x = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # [B*H, W, C]
        elif direction == 'vertical':
            x = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # [B*W, H, C]
        else:
            raise NotImplementedError("Only 'horizontal' and 'vertical' supported.")

        x = x.permute(0, 2, 1).contiguous()  # [B*, C, T]
        params = self.param_gen(x)  # [B*, 4C, T]
        A, B_, C_, D = params.chunk(4, dim=1)  # [B*, C, T] each
        A, B_, C_, D = torch.sigmoid(A), torch.sigmoid(B_), torch.sigmoid(C_), torch.sigmoid(D)

        h = torch.zeros_like(A[:, :, 0])  # [B*, C]
        outputs = []
        for t in range(x.shape[2]):
            x_t = x[:, :, t]
            a_t = A[:, :, t]
            b_t = B_[:, :, t]
            c_t = C_[:, :, t]
            d_t = D[:, :, t]

            h = a_t * h + b_t * x_t
            y_t = c_t * h + d_t * x_t
            outputs.append(y_t.unsqueeze(2))

        y = torch.cat(outputs, dim=2)  # [B*, C, T]
        y = y.permute(0, 2, 1).contiguous()  # [B*, T, C]

        if direction == 'horizontal':
            y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        elif direction == 'vertical':
            y = y.reshape(B, W, H, C).permute(0, 3, 2, 1)  # [B, C, H, W]

        return y

class SelectiveScan2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.horizontal = SS2D_Mamba(dim)
        self.vertical = SS2D_Mamba(dim)
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x):
        h = self.horizontal.forward_scan(x, direction='horizontal')
        v = self.vertical.forward_scan(x, direction='vertical')
        return self.fuse(torch.cat([h, v], dim=1))

class DAMambaBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.norm1 = LayerNorm(dim_in)
        self.linear1 = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        self.dwconv = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, groups=dim_out)
        self.ss2d = SelectiveScan2D(dim_out)
        self.norm2 = LayerNorm(dim_out)
        self.gate = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 1),
            nn.SiLU()
        )
        self.linear2 = nn.Conv2d(dim_out, dim_out, 1)
        self.final_act = nn.SiLU()

        if dim_in != dim_out:
            self.res_proj = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        residual = self.res_proj(x)
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.dwconv(x)
        x = self.ss2d(x)
        x = self.norm2(x)
        gate = self.gate(x)
        x = x * gate
        x = self.linear2(x)
        x = self.final_act(x)
        return x + residual

class CA_Block_V2_Mamba(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        self.train_image_size = train_image_size

        self.q_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)

        total_input_channels = in_channels + len(norm_disp_range)
        self.reduce_fused = nn.Conv2d(total_input_channels, in_channels, kernel_size=1)
        self.fuse_block = DAMambaBlock(in_channels, in_channels)

    def forward(self, feats, directs, image_shape):
        t_feat = feats[0]
        s_feat = feats[1]

        q = self.q_conv(t_feat)
        k = self.k_conv(s_feat)
        warped_feat = self._get_warped_frame(k, directs, image_shape)

        cost = (q.unsqueeze(2) * warped_feat).sum(dim=1) / (q.shape[1] ** 0.5)
        norm_cost = torch.softmax(cost, dim=1)
        fused_input = torch.cat([t_feat, norm_cost], dim=1)
        fused_input = self.reduce_fused(fused_input)
        fused_output = self.fuse_block(fused_input)

        return fused_output, cost

    def _get_warped_frame(self, x, directs, image_shape):
        i_tetha = torch.zeros(1, 2, 3).to(x.device)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        base_coord = F.affine_grid(i_tetha, [1, 1, x.shape[2], x.shape[3]], align_corners=True).to(x.device)
        zeros = torch.zeros_like(base_coord)
        frame_volume = []
        rel_scale = self.train_image_size[1] / image_shape[3] if self.train_image_size[1] != image_shape[3] else 1

        for ch_idx in range(len(self.norm_disp_range)):
            disp_map = zeros.clone()
            disp_map[..., 0] += self.norm_disp_range[ch_idx] * 2 * rel_scale
            grid_coords = disp_map * directs + base_coord
            warped_frame = F.grid_sample(x, grid_coords, mode='bilinear', padding_mode='border', align_corners=True)
            frame_volume.append(warped_frame.unsqueeze(2))

        return torch.cat(frame_volume, dim=2)
'''
