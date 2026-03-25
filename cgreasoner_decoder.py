import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Normalization utilities
# -----------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x):
        mean = x.mean((2,3), keepdim=True)
        var  = x.var((2,3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:,None,None]*x + self.bias[:,None,None]


# -----------------------------
# Depthwise-Separable Conv Block
# (lightweight spatial mixing)
# -----------------------------
class DSConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


# -----------------------------
# MixFFN block
# -----------------------------
class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm   = LayerNorm2d(dim)
        self.fc1    = nn.Conv2d(dim, hidden_dim, 1)
        self.fc2    = nn.Conv2d(hidden_dim, dim, 1)
        self.act    = nn.GELU()
        # layer-scale (stabilizes)
        self.gamma  = nn.Parameter(torch.ones(1, dim, 1, 1)*0.1)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return shortcut + self.gamma * x


# -----------------------------
# Prompt-Conditioned FiLM
#   Generates per-channel scale/bias from a pooled prompt token
#   to modulate image features cheaply and effectively.
# -----------------------------
class PromptFiLM(nn.Module):
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim*2)  # -> [gamma | beta]
        )

    def forward(self, prompt_token):   # (B, C)
        gb = self.mlp(prompt_token)    # (B, 2C)
        B, _ = gb.shape
        C = gb.shape[1] // 2
        gamma = gb[:, :C].unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        beta  = gb[:,  C:].unsqueeze(-1).unsqueeze(-1) # (B,C,1,1)
        return gamma, beta


# -------------------------------------------------------
# PromptedMaskDecoder
# -------------------------------------------------------
class PromptedMaskDecoder(nn.Module):
    """
    Lightweight SegFormer-ish decoder:
      - Prompt projection to image_dim
      - Prompt pooling -> FiLM (per-channel scale/bias)
      - Depthwise-separable conv + MixFFN backbone
      - Two-stage upsampling head
    Keeps the same inputs/outputs as your previous decoder.
    """
    def __init__(self, prompt_dim=4096, image_dim=256, hidden_dim=512):
        super().__init__()
        self.image_dim = image_dim

        # (1) Prompt projection (token-wise) -> image_dim
        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, image_dim),
            nn.LayerNorm(image_dim)
        )

        # (2) Prompt FiLM (uses pooled token)
        self.film = PromptFiLM(image_dim, hidden=hidden_dim)

        # (3) Lightweight fusion backbone
        self.ds1  = DSConv(image_dim)
        self.mffn1 = MixFFN(image_dim, image_dim * 2)

        # (4) Upsampling head (two steps) with MixFFN
        self.up1  = nn.Sequential(
            nn.ConvTranspose2d(image_dim, image_dim//2, 2, stride=2),
            LayerNorm2d(image_dim//2),
            nn.GELU()
        )
        self.mffn2 = MixFFN(image_dim//2, image_dim)

        self.up2  = nn.Sequential(
            nn.ConvTranspose2d(image_dim//2, image_dim//4, 2, stride=2),
            LayerNorm2d(image_dim//4),
            nn.GELU()
        )

        # Final head
        self.final = nn.Sequential(
            nn.Conv2d(image_dim//4, image_dim//8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(image_dim//8, 1, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image_feat, prompt_feat):
        """
        image_feat:  (B, 256, H, W)  – encoder output
        prompt_feat: (B, T, 4096)    – LLM hidden states
        returns:     (B, 1, H, W)
        """
        B, C, H, W = image_feat.shape
        # (1) Token-wise projection and pool
        P   = self.prompt_proj(prompt_feat)      # (B, T, C)
        p_z = P.mean(dim=1)                      # (B, C)

        # (2) Prompt-conditioned FiLM
        gamma, beta = self.film(p_z)             # (B,C,1,1) each

        # (3) Apply FiLM then lightweight backbone
        x = image_feat * (1 + gamma) + beta      # channel-wise modulation
        x = self.ds1(x)
        x = self.mffn1(x)

        # (4) Upsampling head
        x = self.up1(x)                          # (B, C/2, 2H, 2W)
        x = self.mffn2(x)
        x = self.up2(x)                          # (B, C/4, 4H, 4W)

        # (5) Final mask at encoder scale
        mask = self.final(x)                     # (B, 1, 4H, 4W)
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        return mask