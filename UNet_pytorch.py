import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    PyTorch UNet implementation with 6 levels
    Designed for 512x512 input images

    Architecture:
    - Encoder: 6 downsampling levels (512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8)
    - Bottleneck: at 8x8 resolution
    - Decoder: 6 upsampling levels with skip connections
    """

    def __init__(self, n_channels=1, n_classes=1, n_filters=16):
        """
        Args:
            n_channels: Number of input channels (default: 1 for grayscale)
            n_classes: Number of output classes (default: 1 for binary segmentation)
            n_filters: Base number of filters, will be multiplied at each level
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Initial convolution (no downsampling)
        self.inc = DoubleConv(n_channels, n_filters * 4)

        # Encoder path (6 levels)
        self.down1 = Down(n_filters * 4, n_filters * 8)  # 512 -> 256
        self.down2 = Down(n_filters * 8, n_filters * 16)  # 256 -> 128
        self.down3 = Down(n_filters * 16, n_filters * 32)  # 128 -> 64
        self.down4 = Down(n_filters * 32, n_filters * 64)  # 64 -> 32
        self.down5 = Down(n_filters * 64, n_filters * 128)  # 32 -> 16

        # Bottleneck
        self.down6 = Down(n_filters * 128, n_filters * 256)  # 16 -> 8

        # Decoder path (6 levels with skip connections)
        self.up1 = Up(n_filters * 256, n_filters * 128)  # 8 -> 16
        self.up2 = Up(n_filters * 128, n_filters * 64)  # 16 -> 32
        self.up3 = Up(n_filters * 64, n_filters * 32)  # 32 -> 64
        self.up4 = Up(n_filters * 32, n_filters * 16)  # 64 -> 128
        self.up5 = Up(n_filters * 16, n_filters * 8)  # 128 -> 256
        self.up6 = Up(n_filters * 8, n_filters * 4)  # 256 -> 512

        # Output convolution
        self.outc = nn.Conv2d(n_filters * 4, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # 512x512
        x2 = self.down1(x1)  # 256x256
        x3 = self.down2(x2)  # 128x128
        x4 = self.down3(x3)  # 64x64
        x5 = self.down4(x4)  # 32x32
        x6 = self.down5(x5)  # 16x16
        x7 = self.down6(x6)  # 8x8 (bottleneck)

        # Decoder with skip connections
        x = self.up1(x7, x6)  # 16x16
        x = self.up2(x, x5)  # 32x32
        x = self.up3(x, x4)  # 64x64
        x = self.up4(x, x3)  # 128x128
        x = self.up5(x, x2)  # 256x256
        x = self.up6(x, x1)  # 512x512

        # Output
        logits = self.outc(x)
        output = self.sigmoid(logits)
        return output


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        dice = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )

        return 1 - dice


def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Calculate Dice coefficient for evaluation

    Args:
        y_pred: Predicted segmentation (batch_size, channels, height, width)
        y_true: Ground truth segmentation (batch_size, channels, height, width)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient value
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    intersection = (y_pred * y_true).sum()
    dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

    return dice


# Example usage
if __name__ == "__main__":
    # Create model
    model = UNet(n_channels=1, n_classes=1, n_filters=16)

    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with random input
    x = torch.randn(1, 1, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test loss
    criterion = DiceLoss()
    y_true = torch.randint(0, 2, (1, 1, 512, 512)).float()
    loss = criterion(output, y_true)
    print(f"Loss: {loss.item():.4f}")

    # Calculate dice coefficient
    dice = dice_coefficient(output, y_true)
    print(f"Dice coefficient: {dice.item():.4f}")
