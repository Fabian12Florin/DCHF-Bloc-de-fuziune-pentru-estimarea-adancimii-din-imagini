import torch
import torch.nn as nn
import torch.nn.functional as F

# --- S6 Block (Mamba Core) ---
class S6Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = nn.Parameter(torch.eye(dim))
        self.B = nn.Parameter(torch.randn(dim))
        self.C = nn.Parameter(torch.randn(dim))
        self.D = nn.Parameter(torch.randn(dim))
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, L, C = x.shape
        h = torch.zeros(B, C, device=x.device)
        y = []
        for t in range(L):
            h = torch.matmul(h, self.A) + self.B * x[:, t, :]
            y_t = self.C * h + self.D * x[:, t, :]
            y_t = y_t + self.res_scale * x[:, t, :]
            y.append(y_t.unsqueeze(1))
        return torch.cat(y, dim=1)

# --- SS2D Module (fixed dimensions) ---
class SS2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.s6 = S6Block(dim)

    def forward(self, x):
        B, H, W, C = x.shape

        # Horizontal
        horiz = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
        y_h = self.s6(horiz).reshape(B, W, H, C).permute(0, 2, 1, 3)

        # Vertical
        vert = x.reshape(B * H, W, C)
        y_v = self.s6(vert).reshape(B, H, W, C)

        # Diagonal (reconstructed full map)
        diag_map = torch.zeros(B, H, W, C, device=x.device)
        anti_diag_map = torch.zeros(B, H, W, C, device=x.device)
        diag_len = min(H, W)

        diag_input = torch.stack([x[:, i, i, :] for i in range(diag_len)], dim=1)
        anti_diag_input = torch.stack([x[:, i, W - i - 1, :] for i in range(diag_len)], dim=1)

        diag_processed = self.s6(diag_input)
        anti_diag_processed = self.s6(anti_diag_input)

        for i in range(diag_len):
            diag_map[:, i, i, :] = diag_processed[:, i, :]
            anti_diag_map[:, i, W - i - 1, :] = anti_diag_processed[:, i, :]

        return y_h + y_v + diag_map + anti_diag_map

# --- MD (MambaDepth) Block ---
class MDBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.silu = nn.SiLU()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.ss2d = SS2D(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        shortcut = x
        B, H, W, C = x.shape

        x_norm = self.norm1(x)
        f1 = self.silu(self.linear1(x_norm))

        x_conv = self.dwconv(x_norm.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        f2 = self.silu(x_conv)
        f2 = self.ss2d(f2)

        f2 = self.norm2(f2)
        fused = self.linear2(f1 * f2)

        return shortcut + fused

# --- MambaDepth Encoder ---
class MambaDepthEncoder(nn.Module):
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, base_dim, kernel_size=4, stride=4)
        self.stage1 = nn.Sequential(MDBlock(base_dim), MDBlock(base_dim))
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 2, stride=2)
        self.stage2 = nn.Sequential(MDBlock(base_dim * 2), MDBlock(base_dim * 2))
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 2, stride=2)
        self.stage3 = nn.Sequential(MDBlock(base_dim * 4), MDBlock(base_dim * 4))
        self.down3 = nn.Conv2d(base_dim * 4, base_dim * 8, 2, stride=2)
        self.stage4 = nn.Sequential(MDBlock(base_dim * 8), MDBlock(base_dim * 8))

    def forward(self, x):
        feats = []
        # Stage 1
        x = self.patch_embed(x)  # [B, C, H/4, W/4]
        x = x.permute(0, 2, 3, 1)
        x = self.stage1(x)
        feats.append(x.permute(0, 3, 1, 2))
        # Stage 2
        x = self.down1(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)
        x = self.stage2(x)
        feats.append(x.permute(0, 3, 1, 2))
        # Stage 3
        x = self.down2(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)
        x = self.stage3(x)
        feats.append(x.permute(0, 3, 1, 2))
        # Stage 4
        x = self.down3(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)
        x = self.stage4(x)
        feats.append(x.permute(0, 3, 1, 2))
        return feats
