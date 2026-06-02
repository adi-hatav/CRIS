import torch
import torch.nn as nn
from einops import rearrange

SHIFTED_RELU = True


class ShiftedReLU(nn.Module):
    """f(x) = max(x, -1) implemented as ReLU(x + 1) - 1."""

    def __init__(self, shift: float = -1.0, inplace: bool = True):
        super().__init__()
        self.shift = shift
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x - self.shift) + self.shift


def make_norm_layer(ch, kind="instance"):
    if kind == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    if kind == "group":
        return nn.GroupNorm(num_groups=8, num_channels=ch)
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3, norm="instance"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.bn1 = make_norm_layer(out_channels, norm)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.bn2 = make_norm_layer(out_channels, norm)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.act = (
            ShiftedReLU(shift=-1.0, inplace=True)
            if SHIFTED_RELU
            else nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.dropout(x)
        return self.act(x + shortcut)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, _, h, w = x.shape
        assert h % self.window_size == 0 and w % self.window_size == 0, (
            "Input dimensions must be divisible by window size"
        )

        x = rearrange(
            x,
            "b c (h w1) (w w2) -> (b h w) (w1 w2) c",
            w1=self.window_size,
            w2=self.window_size,
        )
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        x = rearrange(
            x,
            "(b h w) (w1 w2) c -> b c (h w1) (w w2)",
            b=b,
            h=h // self.window_size,
            w=w // self.window_size,
            w1=self.window_size,
            w2=self.window_size,
        )
        return x


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, window_size=4):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, base_filters)
        self.enc2 = ResidualBlock(base_filters, base_filters * 2)
        self.enc3 = ResidualBlock(base_filters * 2, base_filters * 4)
        self.enc4 = ResidualBlock(base_filters * 4, base_filters * 8)
        self.enc5 = ResidualBlock(base_filters * 8, base_filters * 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = SwinTransformerBlock(
            dim=base_filters * 16, num_heads=8, window_size=window_size
        )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_filters * 16, base_filters * 16, kernel_size=3, padding=1),
        )
        self.dec5 = ResidualBlock(base_filters * 32, base_filters * 16)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_filters * 16, base_filters * 8, kernel_size=3, padding=1),
        )
        self.dec4 = ResidualBlock(base_filters * 16, base_filters * 8)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
        )
        self.dec3 = ResidualBlock(base_filters * 8, base_filters * 4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
        )
        self.dec2 = ResidualBlock(base_filters * 4, base_filters * 2)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1),
        )
        self.dec1 = ResidualBlock(base_filters * 2, base_filters)
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        bottleneck = self.bottleneck(self.pool(enc5))
        dec5 = self.dec5(torch.cat((self.up5(bottleneck), enc5), dim=1))
        dec4 = self.dec4(torch.cat((self.up4(dec5), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.up3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.up2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), dim=1))
        return self.tanh(self.final_conv(dec1))
