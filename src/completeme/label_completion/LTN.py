import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import SimpleITK as sitk
from pathlib import Path
import gc # Import garbage collection module
from completeme.label_completion.transform_to_volume import get_largest_cc
# network
class UNet3d(nn.Module):
    def contracting_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            #torch.nn.Sigmoid()
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet3d, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channel, 16, 32)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=3, in_channels=128, out_channels=128, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(128),
            torch.nn.Conv3d(kernel_size=3, in_channels=128, out_channels=256, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        )
        # Decode
        self.conv_decode3 = self.expansive_block(128+256, 128, 128)
        self.conv_decode2 = self.expansive_block(64+128, 64, 64)
        self.final_layer = self.final_block(32+64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=False)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=False)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=False)
        final_layer = self.final_layer(decode_block1)
        return final_layer

def eval_model(
    model_file: Path,
    csv_path: Path,
    output_ltn: Path,
    out_channel: int,
    in_channel: int,
) -> None:
    """
    Run inference with the Label Completion Network on a folder of one-hot
    encoded NIfTI volumes and save the post-processed label predictions.

    For each input volume, the model predicts a dense multi-label segmentation.
    Post-processing keeps only the largest connected component per label.

    Args:
        model_file: Path to the model checkpoint (.pth).
        csv_path: Directory containing the '3d_sparse_oh.csv' file listing inputs.
        output_ltn: Directory to save predicted label NIfTI files.
        out_channel: Number of output channels (label classes).
        in_channel: Number of input channels (one-hot labels).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model once — keep in eval mode and disable gradients globally
    unet = UNet3d(in_channel=in_channel, out_channel=out_channel)
    unet.load_state_dict(
        torch.load(model_file, map_location=device)["model_state_dict"]
    )
    unet.to(device).eval()

    image_paths = [
        Path(p) for p in pd.read_csv(csv_path / "3d_sparse_oh.csv")["img"]
    ]

    # Precompute label indices once
    label_indices = np.arange(out_channel)

    with torch.no_grad():
        for path in image_paths:
            # Load and preprocess
            sitk_img = sitk.ReadImage(str(path))

            # Working
            arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # (D, H, W, C)
            tensor = (
                torch.from_numpy(arr)
                .permute(3, 0, 1, 2)       # (C, D, H, W)
                .unsqueeze(0)              # (1, C, D, H, W)
                .to(device)
            )

            # Inference — full precision
            output_np = unet(tensor).squeeze(0).cpu().numpy()  # (C, D, H, W)
            del tensor

            # Post-process: argmax then largest CC per label
            label_pred = np.argmax(output_np, axis=0)  # (D, H, W)
            new_label = np.zeros_like(label_pred, dtype=np.uint8)
            for k in label_indices:

                if k == 0 or k > 8:
                    continue
                seg = label_pred == k
                if seg.any():
                    new_label[get_largest_cc(seg)] = k

            # Save
            result_img = sitk.GetImageFromArray(new_label.astype(np.uint8))
            result_img.CopyInformation(sitk_img)
            sitk.WriteImage(result_img, str(output_ltn / path.name))