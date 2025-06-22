import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoders.submoduleACV import *

# Prima modificare. Folosirea a mai multor capete de atentie. (AFB = attention fusion block)
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
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

        return x, None

# A doua modificare. GatedMLP pentru fuziune adaptiva a canalelor
class GatedMLP(nn.Module):
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

    def forward(self, x):
        x_norm = self.norm(x)
        gate = torch.sigmoid(self.gate_proj(x_norm))
        value = self.value_proj(x_norm)
        gated = gate * value
        out = gated + x
        out = self.ffn(out)
        return out

class MFM_GatedMLP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.fuse_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.gated = GatedMLP(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.filter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.cost_conv = nn.Conv2d(in_channels, 49, kernel_size=3, padding=1)

    def forward(self, feats, directs=None, image_shape=None):
        t_feat, s_feat = feats

        fuse_input = torch.cat([t_feat, s_feat], dim=1)
        fused = self.fuse_conv(fuse_input)

        B, C, H, W = fused.shape
        x = fused.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        x = self.gated(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = x + t_feat
        x = self.norm(x)
        fused_feat = self.filter(x)

        cost_3_s = self.cost_conv(fused_feat)  #(B, 49, H, W)
        return fused_feat, cost_3_s

# A treia modificare. Folosirea a doua module GatedMLP pentru scanare orizontala si verticala
class MFM_DoubleGatedMLP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.fuse_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.mamba_h = GatedMLP(in_channels)
        self.mamba_v = GatedMLP(in_channels)

        self.norm = nn.BatchNorm2d(in_channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        self.cost_conv = nn.Conv2d(in_channels, 49, kernel_size=3, padding=1)

    def forward(self, feats, directs=None, image_shape=None):
        t_feat, s_feat = feats  #(B, C, H, W)

        # Concatenare si fuziune
        fuse_input = torch.cat([t_feat, s_feat], dim=1)  #(B, 2C, H, W)
        fused = self.fuse_conv(fuse_input)               #(B, C, H, W)

        B, C, H, W = fused.shape

        # GatedMLP pentru orizontala
        x_h = fused.permute(0, 2, 3, 1).contiguous()  #(B, H, W, C)
        x_h = x_h.view(B * H, W, C)
        out_h = self.mamba_h(x_h).view(B, H, W, C)

        # GatedMLP pentru verticala 
        x_v = fused.permute(0, 3, 2, 1).contiguous()  #(B, W, H, C)
        x_v = x_v.view(B * W, H, C)
        out_v = self.mamba_v(x_v).view(B, W, H, C).permute(0, 2, 1, 3)  # (B, H, W, C)

        # Combinarea rezultatelor
        out = out_h + out_v  # (B, H, W, C)

        # Conexiune de ocolire, normalizare si o retea finala pentruf filtrare
        out = out.permute(0, 3, 1, 2)  #(B, C, H, W)
        out = out + t_feat
        out = self.norm(out)
        fused_feat = self.out_conv(out)

        cost_3_s = self.cost_conv(fused_feat)
        return fused_feat, cost_3_s

# A patra modificare. Mamba ca bloc final; inlocuieste reteaua finala redu
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
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
            x = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        elif direction == 'vertical':
            x = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        else:
            raise NotImplementedError("Only 'horizontal' and 'vertical' supported.")

        x = x.permute(0, 2, 1).contiguous()
        params = self.param_gen(x)
        A, B_, C_, D = params.chunk(4, dim=1)
        A, B_, C_, D = torch.sigmoid(A), torch.sigmoid(B_), torch.sigmoid(C_), torch.sigmoid(D)

        h = torch.zeros_like(A[:, :, 0])  
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

        y = torch.cat(outputs, dim=2) 
        y = y.permute(0, 2, 1).contiguous()

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

class CA_Block_V3_Mamba(nn.Module):
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

# A cincea modificare. Mecanism pentru crearea unui volum de atentie pentru adancime (DAV)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
       from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, in_channels, out_channels, use_refl=True, bias=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=bias)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class CA_Block_V3_DAV(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None, patch_size=1):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        self.train_image_size = train_image_size
        cost_channels = len(norm_disp_range)
 
        # Numar de canale redus
        self.q_conv = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.k_conv = nn.Conv2d(in_channels, in_channels // 4, 1)
 
        # Blocul DAV; face similaritate cosinus pe patchuri; 
        # Datorita resurselor reduse s-a folosit o marime a patchului de 1
        self.dav = DAV_Block(in_channels // 4, patch_size=patch_size)
 
        # Filtru tridimension ca postprocesare
        self.cost_filter = nn.Conv3d(1, 1, kernel_size=(3,3,3), padding=1)
 
        # Normalizare
        self.norm_t = nn.GroupNorm(1, in_channels)
        self.norm_c = nn.GroupNorm(1, cost_channels)
 
        self.cost_expand = nn.Conv2d(cost_channels, in_channels, kernel_size=1)
 
        # Fuziunea adaptiva
        self.fusion_alpha = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, 1),
            nn.Sigmoid()
        )
 
        # Blocul de reducere si activare a non-liniaritatii
        self.redu = nn.Sequential(
            Conv3x3(in_channels, in_channels),
            SELayer(in_channels),
            nn.ELU()
        )
 
    def forward(self, feats, directs, image_shape):
        t_feat = feats[0]
        s_feat = feats[1]
 
        q = self.q_conv(t_feat)
        k = self.k_conv(s_feat)
        warped_feat = self._get_warped_frame(k, directs, image_shape, q.shape[2:])  # [B, C, H, W, D]
 
        # Se construieste volumul de cost
        dav_cost = self.dav(q, warped_feat)  # [B, D, H, W]
 
        # Se aplica filtrul tridimensional si se face o distributie de probabilitate
        # prin functia softmax
        dav_cost_3d = dav_cost.unsqueeze(1)  # [B, 1, D, H, W]
        filtered_cost = self.cost_filter(dav_cost_3d).squeeze(1)  # [B, D, H, W]
        norm_cost = torch.softmax(filtered_cost, dim=1)
 
        # Se normalizeaza rezultatele si se face fuziunea adaptiva
        t_feat_n = self.norm_t(t_feat)
        norm_cost_n = self.norm_c(norm_cost)
        cost_feat = self.cost_expand(norm_cost_n)  # [B, 256, H, W]
        fusion_input = torch.cat([t_feat_n, cost_feat], dim=1)
        alpha = self.fusion_alpha(fusion_input)
        fused = alpha * t_feat_n + (1 - alpha) * cost_feat

        if fused.shape[2:] != t_feat.shape[2:]:
            fused = F.interpolate(fused, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
 
        x = self.redu(fused)
        return x, norm_cost
 
    def _get_warped_frame(self, x, directs, image_shape, target_hw):
        B, C, Hx, Wx = x.shape
        Ht, Wt = target_hw
        D = len(self.norm_disp_range)
 
        i_tetha = torch.zeros(1, 2, 3, device=x.device)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        normal_coord = F.affine_grid(i_tetha, [1, 1, Hx, Wx], align_corners=True).to(x.device)
        base_coord = normal_coord
        zeros = torch.zeros_like(base_coord)
 
        if self.train_image_size is not None and self.train_image_size[1] != image_shape[3]:
            rel_scale = self.train_image_size[1] / image_shape[3]
        else:
            rel_scale = 1
 
        frame_volume = []
        for disp in self.norm_disp_range:
            disp_map = zeros.clone()
            disp_map[..., 0] += disp * 2 * rel_scale
            grid_coords = disp_map * directs + base_coord
            warped = F.grid_sample(x, grid_coords, mode='bilinear', padding_mode='border', align_corners=True)
            frame_volume.append(warped.unsqueeze(2))
 
        warped_feat = torch.cat(frame_volume, dim=2)  # [B, C, D, Hx, Wx]
        warped_feat = warped_feat.permute(0, 1, 3, 4, 2)  # [B, C, Hx, Wx, D]
 
        if (Hx, Wx) != (Ht, Wt):
            warped_feat = warped_feat.permute(0, 4, 1, 2, 3)  # [B, D, C, Hx, Wx]
            warped_feat = warped_feat.reshape(B * D, C, Hx, Wx)
            warped_feat = F.interpolate(warped_feat, size=(Ht, Wt), mode='bilinear', align_corners=False)
            warped_feat = warped_feat.view(B, D, C, Ht, Wt).permute(0, 2, 3, 4, 1)
 
        return warped_feat
 
class DAV_Block(nn.Module):
    def __init__(self, in_channels, patch_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2)
 
    def forward(self, q, warped_feat):
        B, C, H, W = q.shape
        D = warped_feat.shape[-1]
        P = self.patch_size
 
        q_patches = self.unfold(q)  # [B, C*P*P, H*W]
        q_patches = F.normalize(q_patches, dim=1)
        q_patches = q_patches.permute(0, 2, 1)  # [B, H*W, F]
 
        k_patches = []
        for d in range(D):
            k = warped_feat[..., d]  # [B, C, H, W]
            k_patch = self.unfold(k)  # [B, C*P*P, H*W]
            k_patch = F.normalize(k_patch, dim=1)
            k_patch = k_patch.permute(0, 2, 1).unsqueeze(-1)  # [B, H*W, F, 1]
            k_patches.append(k_patch)
        k_patches = torch.cat(k_patches, dim=-1)  # [B, H*W, F, D]
 
        sim = torch.einsum('bnf,bnfd->bnd', q_patches, k_patches)  # [B, H*W, D]
        sim = sim.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        return sim

# A sasea modificare. Folosirea atentiei pentru a capta context dual in modulul de fuziune
class DCW_Block(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None, reduction_factor=4):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        self.train_image_size = train_image_size
        self.max_disp = len(norm_disp_range)
        
        # Reducere a dimensiunii canalelor
        reduced_channels = in_channels // reduction_factor
        
        # Reducere de dimensiune pentru q și k
        self.q_conv = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, reduced_channels),
            nn.ELU()
        )
        
        self.k_conv = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, reduced_channels),
            nn.ELU()
        )
        
        # Ramura contextului global
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(reduced_channels, reduced_channels // 2, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(reduced_channels // 2, reduced_channels, kernel_size=1)
        )
        
        # Ramura contextului local
        self.local_attn = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, reduced_channels),
            nn.ELU()
        )
        
        # Blocul de fuziune a atentiei
        self.attn_fusion = nn.Sequential(
            nn.Conv2d(reduced_channels * 2, reduced_channels, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(reduced_channels, self.max_disp, kernel_size=1)
        )
        
        # Blocul de rafinare a volumului de cost
        self.refine_cost = nn.Sequential(
            nn.Conv2d(self.max_disp, self.max_disp, kernel_size=3, padding=1, groups=self.max_disp),
            nn.ELU(),
            nn.Conv2d(self.max_disp, self.max_disp, kernel_size=1)
        )
        
        # Blocul de fuziune finala
        self.fusion = nn.Sequential(
            Conv3x3(in_channels + self.max_disp, in_channels),
            SELayer(in_channels),
            nn.ELU()
        )

    def forward(self, feats, directs, image_shape):
        left_feat, right_feat = feats[0], feats[1]
        
        q = self.q_conv(left_feat)
        k = self.k_conv(right_feat)
        
        global_feat = self.global_attn(q)
        local_feat = self.local_attn(q)
        
        global_feat = global_feat.expand(-1, -1, q.shape[2], q.shape[3])
        
        combined_feat = torch.cat([global_feat, local_feat], dim=1)
        attn_weights = self.attn_fusion(combined_feat)
        
        warped_volume = self._get_attention_warped_features(
            k, attn_weights, directs, image_shape, q.shape[2:]
        )
        
        cost = self.refine_cost(warped_volume)
        norm_cost = torch.softmax(cost, dim=1)
        
        x = self.fusion(torch.cat([left_feat, norm_cost], dim=1))
        
        return x, norm_cost
    
    def _get_attention_warped_features(self, x, attention_weights, directs, image_shape, target_hw):
        B, C, Hx, Wx = x.shape
        Ht, Wt = target_hw
        D = len(self.norm_disp_range)
        
        warped_volume = torch.zeros((B, D, Ht, Wt), device=x.device)
      
        i_tetha = torch.zeros(1, 2, 3, device=x.device)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        base_grid = F.affine_grid(i_tetha, [1, 1, Hx, Wx], align_corners=True)
        base_grid = base_grid.repeat(B, 1, 1, 1)
        zeros = torch.zeros_like(base_grid)

        if self.train_image_size is not None and self.train_image_size[1] != image_shape[3]:
            rel_scale = self.train_image_size[1] / image_shape[3]
        else:
            rel_scale = 1
        
        chunk_size = min(8, D)
        for d_start in range(0, D, chunk_size):
            d_end = min(d_start + chunk_size, D)
            chunk_disp = self.norm_disp_range[d_start:d_end]
            
            for idx, disp in enumerate(chunk_disp):
                d_idx = d_start + idx
                
                disp_map = zeros.clone()
                disp_map[..., 0] = disp_map[..., 0] + disp * 2 * rel_scale
                
                grid_coords = disp_map * directs.view(-1, 1, 1, 1) + base_grid
                
                warped_feat = F.grid_sample(
                    x, grid_coords, 
                    mode='bilinear', 
                    padding_mode='border', 
                    align_corners=True
                )
                
                attn_weight = attention_weights[:, d_idx:d_idx+1]
                if warped_feat.shape[2:] != attn_weight.shape[2:]:
                    warped_feat = F.interpolate(
                        warped_feat, 
                        size=attn_weight.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                weighted_feat = torch.mean(warped_feat, dim=1, keepdim=True)
                weighted_feat = weighted_feat * F.sigmoid(attn_weight)
                
                warped_volume[:, d_idx] = weighted_feat.squeeze(1)
        
        return warped_volume
    
# A saptea modificare. CosineWarpAttentionFusion
# Folosirea similaritatii cosinus pe hartile de disparitate si fuziune adaptiva
class CWAF_Block(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        self.train_image_size = train_image_size
        self.max_disp = len(norm_disp_range)
        reduced_channels = in_channels // 4

        # Reducerea canalelor inainte de filtrare
        self.q_conv = nn.Conv2d(in_channels, reduced_channels, 1)
        self.k_conv = nn.Conv2d(in_channels, reduced_channels, 1)

        # Se asigura ca forma volumui de cost este potrivita
        self.cost_expand = nn.Conv2d(reduced_channels, in_channels, 1)

        # Calculul parametrului alfa, folosit in fuziuena adaptiva
        self.fusion_alpha = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, 1),
            nn.Sigmoid()
        )

        # Bloc final de reducere si activare a non-liniaritatii
        self.redu = nn.Sequential(
            Conv3x3(in_channels, in_channels, use_refl=True, bias=False),
            SELayer(in_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, feats, directs, image_shape):
        t_feat, s_feat = feats
        B, _, H, W = t_feat.shape
        D = self.max_disp

        # Canalele sunt reduse si normalizate
        q = F.normalize(self.q_conv(t_feat), dim=1)
        k = F.normalize(self.k_conv(s_feat), dim=1)

        # Se construieste harta de disparitati
        warped_k = self._get_warped_features(k, directs, image_shape, (H, W))  # [B, C, H, W, D]

        # Similaritate cosinus
        sim = (q.unsqueeze(-1) * warped_k).sum(1)  # [B, H, W, D]
        sim = sim.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

        # Se foloseste softmax pentru crearea unei distributii de probabilitate
        prob = torch.softmax(sim, dim=1)  # [B, D, H, W]

        # Dimensiunile sunt aliniate
        prob_perm = prob.permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, H, W, D]

        # Suma peste disparitati
        cost_feat = (warped_k * prob_perm).sum(-1)  # [B, C, H, W]

        # Se extinde volumul de cost pentru a se asigura forma corecta
        cost_feat_exp = self.cost_expand(cost_feat)  # [B, C, H, W]

        # Fuziunea adaptiva
        fusion_input = torch.cat([t_feat, cost_feat_exp], dim=1)
        alpha = self.fusion_alpha(fusion_input)
        fused = alpha * t_feat + (1 - alpha) * cost_feat_exp

        # Bloc de reducere
        out = self.redu(fused)

        return out, prob

    def _get_warped_features(self, x, directs, image_shape, target_hw):
        B, C, Hx, Wx = x.shape
        Ht, Wt = target_hw
        D = self.max_disp

        i_theta = torch.zeros(1, 2, 3, device=x.device)
        i_theta[:, 0, 0] = 1
        i_theta[:, 1, 1] = 1
        base_grid = F.affine_grid(i_theta, [1, 1, Hx, Wx], align_corners=True).to(x.device)
        zeros = torch.zeros_like(base_grid)

        rel_scale = (self.train_image_size[1] / image_shape[3]) if self.train_image_size and self.train_image_size[1] != image_shape[3] else 1

        warped_list = []
        for disp in self.norm_disp_range:
            disp_map = zeros.clone()
            disp_map[..., 0] += disp * 2 * rel_scale
            coords = disp_map * directs + base_grid
            warped = F.grid_sample(x, coords, mode='bilinear', padding_mode='border', align_corners=True)
            warped_list.append(warped.unsqueeze(-1))  # [B, C', H, W, 1]

        warped_stack = torch.cat(warped_list, dim=-1)  # [B, C', H, W, D]
        return warped_stack

# A opta modificare. Bloc de fuziune a volumului corelatiei atentiei
class ACV_Block(nn.Module):
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
        att_weights = F.softmax(att_weights, dim=2)  # softmax pe disparitati

        cost = att_weights.squeeze(1)  # [B, D, H, W]

        # Partea stanga si cost sunt proiectate pentru a avea dimensiunea corecta
        left_proj = self.left_proj(left) if left.shape[1] == self.left_proj.in_channels else \
            nn.Conv2d(left.shape[1], self.left_proj.out_channels, 1, device=left.device, dtype=left.dtype)(left)
        cost_proj = self.cost_proj(cost) if cost.shape[1] == self.cost_proj.in_channels else \
            nn.Conv2d(cost.shape[1], self.cost_proj.out_channels, 1, device=cost.device, dtype=cost.dtype)(cost)

        x = self.redu(torch.cat([left_proj, cost_proj], dim=1))
        return x, cost

# A noua modificare. Metrica Mahalanobis invatabila
class HMF_Block(nn.Module):
    def __init__(self, in_channels, norm_disp_range, train_image_size=None):
        super().__init__()
        self.norm_disp_range = norm_disp_range
        self.train_image_size = train_image_size
        self.D = len(norm_disp_range)
        self.C_mid = in_channels // 4

        # Reducere a numarului de canale
        self.q_conv = nn.Conv2d(in_channels, self.C_mid, 1)
        self.k_conv = nn.Conv2d(in_channels, self.C_mid, 1)

        # Metrica Mahalanobis invatabila. Matrice simetrica
        self.metric = nn.Parameter(torch.eye(self.C_mid))

        # Activare kernel (tanh(alpha·sim + beta))
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # Filtrare 3D
        self.cost_filter = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Atentie pe disparitate
        self.disp_att = nn.Sequential(
            nn.Conv2d(self.C_mid * 2, self.D, 1),
            nn.Sigmoid()
        )

        # Operatie de extindere pentru potrivirea dimensiunii
        self.expand = nn.Conv2d(self.D, in_channels, 1)

        # Calculul parametrului alfa pentru fuziunea adaptiva
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, 1),
            nn.Sigmoid()
        )

        # Reducere finala
        self.redu = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            SELayer(in_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, feats, directs, image_shape):
        t_feat, s_feat = feats
        B, _, H, W = t_feat.shape

        q = F.normalize(self.q_conv(t_feat), dim=1)  # [B, C', H, W]
        k = F.normalize(self.k_conv(s_feat), dim=1)  # [B, C', H, W]

        # Warp k la fiecare disparitate
        warped_k = self._get_warped_frame(k, directs, image_shape, q.shape[2:])  # [B, C', H, W, D]

        # Similaritate cu metrica Mahalanobis
        M = 0.5 * (self.metric + self.metric.T)
        q_proj = torch.einsum('bchw,cd->bdhw', q, M).unsqueeze(-1)  # [B, C', H, W, 1]
        sim = (q_proj * warped_k).sum(1, keepdim=True)  # [B, 1, H, W, D]
        sim = torch.tanh(self.alpha * sim + self.beta)

        # Volum de cost filtrat 3D
        C = sim.permute(0, 1, 4, 2, 3).contiguous()     # [B, 1, D, H, W]
        C = self.cost_filter(C).squeeze(1)              # [B, D, H, W]

        # Atentie pe disparitate
        k_mean = warped_k.mean(dim=-1)  # [B, C', H, W]
        disp_att = self.disp_att(torch.cat([q, k_mean], dim=1))  # [B, 2*C', H, W]
        C = C * disp_att  # atentie pe disparitate

        # Softmax pentru crearea unei distributii de probabilitate
        P = torch.softmax(C, dim=1)  # [B, D, H, W]

        # Extinderea volumului de cost
        cost_feat = self.expand(P)  # [B, C, H, W]

        # Fuziune adaptiva
        fusion_input = torch.cat([t_feat, cost_feat], dim=1)
        alpha = self.fusion_gate(fusion_input)
        fused = alpha * t_feat + (1 - alpha) * cost_feat

        x = self.redu(fused)
        return x, P  # [B, C, H, W], [B, D, H, W]

    def _get_warped_frame(self, x, directs, image_shape, target_hw):
        B, C, Hx, Wx = x.shape
        Ht, Wt = target_hw
        D = len(self.norm_disp_range)

        i_theta = torch.zeros(1, 2, 3, device=x.device)
        i_theta[:, 0, 0] = 1
        i_theta[:, 1, 1] = 1
        base_coord = F.affine_grid(i_theta, [1, 1, Hx, Wx], align_corners=True).to(x.device)
        zeros = torch.zeros_like(base_coord)

        rel_scale = (self.train_image_size[1] / image_shape[3]) if self.train_image_size and self.train_image_size[1] != image_shape[3] else 1

        volume = []
        for disp in self.norm_disp_range:
            disp_map = zeros.clone()
            disp_map[..., 0] += disp * 2 * rel_scale
            coords = disp_map * directs + base_coord
            warped = F.grid_sample(x, coords, mode='bilinear', padding_mode='border', align_corners=True)
            volume.append(warped.unsqueeze(2))
        feat = torch.cat(volume, dim=2).permute(0, 1, 3, 4, 2).contiguous()
        return feat  # [B, C, H, W, D]

# Solutia propusa. Folosirea a doua volume de cost, a filtrarii hourglass si
# a atentiei pe canale, impreuna cu fuziune adaptiva
class DCHF_Block(nn.Module):
    def __init__(self, in_channels, disp_range, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.disp_range = disp_range
        self.maxdisp = len(disp_range)

        # Se reduce numarul de canale
        self.q_conv = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.k_conv = nn.Conv2d(in_channels, in_channels // 4, 1)

        # Se declara metoda hourglass din submodulul submoduleACV.py
        self.hourglass = hourglass_att(2)  # 2 canale pentru concat de volume

        # Atentie pe canale pentru iesirea de la hourglass
        self.channel_att = channelAtt(1, in_channels // 4)

        # Operatie de extindere a dimensiunii pentru a pastra forma corecta
        self.cost_expand = nn.Conv2d(self.maxdisp, in_channels, 1)

        # Calculul parametrului alfa pentru fuziune adaptiva
        self.fusion_alpha = nn.Sequential(
            nn.Conv2d(in_channels*2, 1, 1),
            nn.Sigmoid()
        )

        # Bloc final de reducere si activare a non-liniaritatii
        self.redu = nn.Sequential(
            Conv3x3(in_channels + self.maxdisp, in_channels),
            SELayer(in_channels, reduction),
            nn.ELU()
        )

    def forward(self, feats, directs, image_shape):
        t_feat, s_feat = feats
        q = self.q_conv(t_feat)
        k = self.k_conv(s_feat)

        # Se construiesc doua volume de cost care urmeaza a fi concatenate
        cost_corr = build_norm_correlation_volume(q, k, self.maxdisp)  # [B, 1, D, H, W]
        cost_abs = build_absdiff_volume(q, k, self.maxdisp)            # [B, 1, D, H, W]
        cost_volume = torch.cat([cost_corr, cost_abs], dim=1)          # [B, 2, D, H, W]

        # Piramida de caracteristici formata din t_feat la diferite rezolutii
        imgs = [t_feat, F.avg_pool2d(t_feat, 2), F.avg_pool2d(t_feat, 4), F.avg_pool2d(t_feat, 8)]
        
        cost_volume = self.hourglass(cost_volume, imgs)                # [B, 1, D, H, W]
        cost_volume = self.channel_att(cost_volume, q)                 # [B, 1, D, H, W]
        cost_volume = cost_volume.squeeze(1)                           # [B, D, H, W]
        
        # Softmax pentru a obtine distrubutia de probabilitate
        norm_cost = torch.softmax(cost_volume, dim=1)                  # [B, D, H, W]

        cost_feat = self.cost_expand(norm_cost)                        # [B, C, H, W]

        fusion_input = torch.cat([t_feat, cost_feat], dim=1)
        alpha = self.fusion_alpha(fusion_input)
        fused = alpha * t_feat + (1 - alpha) * cost_feat

        x = torch.cat([fused, norm_cost], dim=1)           # [B, C+D, H, W]
        out = self.redu(x)                                 # [B, C, H, W]

        return out, norm_cost