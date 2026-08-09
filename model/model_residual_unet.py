# model_residual_unet.py
import time
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from loss_vgg import PerceptualLoss
from activations import get_activation

# Residual U-Net with Multi-Path Residual Feature Blocks
#
# This model is designed specifically for image enhancement of the FS-UAE Amiga emulator.
# FS-UAE uses a single frame buffer with 32-bit RGBA format, where each color channel is in sRGB space.
# The model is trained in sRGB space, to be compatible with VGG.
# 
# The FS-UAE framebuffer has a fixed size of 752x576 pixels.
# Amiga lores mode uses 4 raw pixels for every displayed pixel, in a 2x2 grid, resulting in a 376x288 effective resolution.
# Amiga hires mode uses 2 raw pixels for every displayed pixel, in a 1x2 grid, resulting in a 752x288 effective resolution.
# Amiga lores interlace mode uses 2 raw pixels for every displayed pixel, in a 2x1 grid, resulting in a 376x576 effective resolution.
# Amiga hires interlace mode uses 1 raw pixel for every displayed pixel, in a 1x1 grid, resulting in a 752x576 effective resolution.
# 
# Amiga superresolution modes (e.g., Super HiRes) are not supported by FS-UAE and thus not considered here.
#
# The model uses a Residual U-Net architecture with multi-path residual feature blocks.
# The input is processed through a head module that combines pixel unshuffling, pooling, and
# a new convolutional path to extract rich features before entering the U-Net.
# The U-Net consists of an encoder-decoder structure with skip connections and a bottleneck.
# The output is refined and combined with a global skip connection from the input.
# The model is trained using a perceptual loss that combines pixel-wise loss, VGG-based perceptual loss, and high-frequency loss.
# 
# Amiga games are mostly handdrawn pixel art with sharp edges and limited color palettes.
# Often artists used dithering to simulate more colors and gradients, but not always in a consistent way.
# The model is trained on a dataset of high resolution images (natural images but mostly game footage)
# in full colour with there corresponding lores counterparts.
# 
# Dithering patterns applied to lores images in the training dataset include:
# - Floyd-Steinberg dithering
# - Bayer ordered dithering '2x2', '4x4', '8x8'
# - Atkinson dithering
# - Sierra dithering '2', '3'
# - Stucki dithering
# - Burkes dithering
# - Checkerboard dithering 
# - No dithering
#
# Colour modes in the training dataset include:
# - 16 color palette
# - 24 color palette 
# - 32 color palette
# - 64 color paletted extra half-brite
# - 128 color palette
# - 256 color palette
# - 512 color palette
# - HAM6 for Amiga lores and lores interlace
# - SHAM (Split HAM6) for Amiga lores and lores interlace
#
# We train models for two case, OCS and AGA.
# OCS is the original Amiga chipset, which has a maximum of 32 colors on screen at once (with some tricks to get more).
# AGA is the Advanced Graphics Architecture, which has a maximum of 256 colors on screen at once.
# OCS games are typically more pixelated and have more dithering, while AGA games can have smoother gradients and more colors.
# The OCS dataset consists out of the following modes: palette 16 24 32 64 128, EHB, HAM6, SHAM.
# The AGA dataset consists out of the following modes: palette 0 128 256 512 (0 meaning full 24 bit color).

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True), # Optimized: ReLU is faster
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """
    Standard Residual Block (Conv-ReLU-Conv).
    """
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3,
                 activation1='relu', activation2='relu',
                 activation1_params=None, activation2_params=None,
                 inplace=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding=kernel_size//2, bias=True)
        # Build activations, forwarding any provided params and ensuring
        # channel-aware activations get a sensible default if params omitted.
        def make_act(name, channels_for_act, params, inplace_local=False):
            if name is None:
                return nn.Identity()
            key = name.lower()
            # Copy params to avoid mutating caller dict
            p = dict(params) if params is not None else {}

            # Provide sensible defaults for activations that expect channel counts
            if key in ('apprelu', 'app_relu', 'aprelu') and 'channels' not in p:
                p['channels'] = channels_for_act
            if key in ('conv_relu', 'convrelu') and 'channels' not in p:
                p['channels'] = channels_for_act
            if key in ('biased_relu', 'biasedprelu', 'biased_prelu', 'biased') and 'num_parameters' not in p:
                p['num_parameters'] = channels_for_act
            if key == 'prelu' and 'num_parameters' not in p:
                print(f"Warning: PReLU activation in ResidualBlock with {channels_for_act} channels but 'num_parameters' not specified. Defaulting to num_parameters={channels_for_act}. For better performance, consider explicitly setting 'num_parameters' in the activation parameters.")
                p['num_parameters'] = channels_for_act

            # Ensure inplace is passed when requested and constructor accepts it
            if inplace_local and 'inplace' not in p:
                p['inplace'] = True

            try:
                return get_activation(name, params=p, inplace=inplace_local)
            except TypeError as e:
                print(f"Error occurred while creating activation {name}: {e}")
                # Fall back: try without params
                return get_activation(name, params=None, inplace=inplace_local)

        #print(f"ResidualBlock: activation1={activation1} with params {activation1_params}, activation2={activation2} with params {activation2_params}")
        #print(f"ResidualBlock: conv1 in_channels={in_channels}, mid_channels={mid_channels}, out_channels={out_channels}, kernel_size={kernel_size}")
        self.act1 = make_act(activation1, mid_channels, activation1_params, inplace_local=inplace)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size, padding=kernel_size//2, bias=True)
        self.act2 = make_act(activation2, out_channels, activation2_params, inplace_local=inplace) # Post-addition activation

        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)

        if self.skip_proj:
            residual = self.skip_proj(residual)

        x = residual + 0.1 * x  # Scaled residual for stability
        x = self.act2(x) # Post-addition activation
        return x

class kPathResidualFeatureBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 k_paths=3, with_skip_connection=True,
                 use_concatenation=True, use_attention=False, use_merge=True):
        super().__init__()

        self.k_paths = k_paths
        self.with_skip_connection = with_skip_connection
        self.use_concatenation = use_concatenation
        self.use_attention = use_attention
        self.use_merge = use_merge
        self.skip_alpha = None
        self.skip_proj = None

        self.path_acts = nn.ModuleList([
            get_activation("mish", inplace=True),                                               # Path 0: Smooth gradients
            get_activation("prelu", params={"num_parameters": mid_channels}, inplace=False),    # Path 1: Negative residuals
            get_activation("telu"),                                                             # Path 2: Text/UI edges
            get_activation("biased_relu", params={"num_parameters": mid_channels})              # Path 4: Color offsets
        ])

        # Build multi-path feature extraction blocks
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
                self.path_acts[i % len(self.path_acts)],
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            )
            for i in range(k_paths)
        ])

        # Combine features from paths
        combined_channels = out_channels * k_paths if use_concatenation else out_channels
        final_out_channels = out_channels if use_merge else combined_channels

        if use_merge:
            self.merge_conv = nn.Sequential(
                nn.Conv2d(combined_channels, out_channels, kernel_size=1, bias=True)
            )

        # Optional attention
        if use_attention:
            self.attention = SqueezeExcite(combined_channels)

        if with_skip_connection:
            if in_channels != final_out_channels:
                self.skip_proj = nn.Conv2d(in_channels, final_out_channels, kernel_size=1, bias=False)
            else:
                self.skip_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).float()

    def forward(self, x):
        # Parallel paths
        path_outputs = [path(x) for path in self.paths]

        # Aggregate
        if self.use_concatenation:
            combined = torch.cat(path_outputs, dim=1)
        else:
            combined = sum(path_outputs) / self.k_paths

        # Optional attention
        if self.use_attention:
            combined = self.attention(combined)

        # Merge
        if self.use_merge:
            fused = self.merge_conv(combined)
        else:
            fused = combined

        # Residual
        if self.with_skip_connection:
            skip = self.skip_proj(x) if self.skip_proj else x
            if self.skip_alpha is not None:
                skip = skip * self.skip_alpha
            fused = fused + skip

        return fused

class DecayedFastMambaBlock(nn.Module):
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.d_inner = channels * 2
        self.d_state = d_state

        self.in_proj = nn.Linear(channels, self.d_inner * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, channels, bias=False)

        # Learnable decay factor for the row context
        self.decay_factor = nn.Parameter(torch.ones(1, 1, d_state))

    def forward(self, x):
        # Force execution within local autocast context to keep MatMul ops in FP16
        with torch.autocast(device_type=x.device.type, enabled=True, dtype=torch.float16):
            orig_dtype = x.dtype

            # Keep accumulator-sensitive scaling/decay weights in FP32, cast inputs for the MatMul block
            decay_fp32 = self.decay_factor.to(torch.float32)

            # 1. Handle 4D input: [B, C, H, W] -> [B*H, W, C]
            is_4d = len(x.shape) == 4
            if is_4d:
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1).reshape(B * H, W, C)

            # x shape is now [Batch, Seq, Channels]
            xz = self.in_proj(x)
            x_proj, z = xz.chunk(2, dim=-1)

            # SSM States (Key/Query/Value approximation)
            states = self.x_proj(x_proj) 
            B_ssm, C_ssm = states.chunk(2, dim=-1)

            # Decayed Global Aggregation (MatMul executed in FP16 via autocast)
            gate = torch.sigmoid(B_ssm)

            # Promote only the decay multiplication step to FP32 to prevent underflow, then cast back to FP16 for MatMul
            context_gate = (gate * decay_fp32).to(x_proj.dtype)
            context = torch.matmul(context_gate.transpose(-1, -2), x_proj)

            # Apply memory and gate (FP16 MatMul)
            y = torch.matmul(C_ssm, context)
            y = y * F.silu(z)
            out = self.out_proj(y)

            # 2. Restore 4D shape: [B*H, W, C] -> [B, C, H, W]
            if is_4d:
                out = out.view(B, H, W, -1).permute(0, 3, 1, 2)

            out = out.to(orig_dtype)

        return out

class CrossScanMambaBottleneck(nn.Module):
    """Issue: when attaching multiple cross-scan Mamba blocks, MiGraphX fails to compile."""

    def __init__(self, channels, d_state = 16):
        super().__init__()
        # Path 1: Horizontal Scan (for text/rows)
        self.mamba_h = DecayedFastMambaBlock(channels, d_state)

        # Path 2: Vertical Scan (for Copper/gradients)
        self.mamba_v = DecayedFastMambaBlock(channels, d_state)

        # Path 3: Spatial (for dither patterns)
        self.spatial = nn.Sequential(
            # 1. Look at local 3x3 context (Depthwise = Cheap)
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True), # Use SiLU/Swish for smoother gradients
            # 2. Project back (Pointwise = Mixes information)
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )

        # Merge: 4 paths (Identity + H + V + Spatial)
        self.selector = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Horizontal Path
        y_h = self.mamba_h(x)

        # 2. Vertical Path
        # Swap H and W so the 'sequence' is the column
        x_v = x.transpose(2, 3) # [B, C, W, H]
        y_v = self.mamba_v(x_v)
        del x_v
        y_v = y_v.transpose(2, 3) # [B, C, H, W]

        # 3. Spatial Path
        y_s = self.spatial(x)

        # 4. Gated Concatenation
        feat = torch.cat([x, y_h, y_v, y_s], dim=1)
        out = self.selector(feat)

        return self.norm(out)


class HeadProcessing(nn.Module):
    def __init__(self, input_channels, base_channels, k_paths, onebyone_expansion=2.0, twobytwo_expansion=2.0, preprocessing=True, with_residual=True):
        super().__init__()

        self.preprocessing = preprocessing
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        unshuffled_ch = input_channels * 4

        if preprocessing:
            self.avg_pool = nn.AvgPool2d(2, stride=2)
            self.max_pool = nn.MaxPool2d(2, stride=2)

            self.conv_2x2_path = nn.Conv2d(
                in_channels=input_channels,
                out_channels=int(input_channels * twobytwo_expansion),
                kernel_size=2,
                stride=2
            )

            self.conv_expand = nn.Conv2d(
                in_channels=unshuffled_ch,
                out_channels=int(unshuffled_ch * onebyone_expansion),
                kernel_size=1,
                padding=0,
                stride=1
            )

            concat_input_channels = unshuffled_ch + \
                        int(unshuffled_ch * onebyone_expansion) + \
                        (3 * input_channels) + \
                        int(input_channels * twobytwo_expansion)
        else:
            # In lores_only, we only pass the unshuffled 4x channels
            concat_input_channels = unshuffled_ch

        self.conv_stack = kPathResidualFeatureBlock(
                in_channels=concat_input_channels,
                mid_channels=int(base_channels * 1.5),
                out_channels=base_channels,
                k_paths=k_paths,
                with_skip_connection=False,
                use_concatenation=True,
                use_attention=False,
                use_merge=False
            )

        stack_out_ch = base_channels * k_paths
        se_in_ch = stack_out_ch + unshuffled_ch
        conv_reduce_in = stack_out_ch + unshuffled_ch

        self.se = SqueezeExcite(se_in_ch, reduction=8)
        self.conv_proj = nn.Conv2d(unshuffled_ch, base_channels, kernel_size=1, bias=True)
        self.conv_reduce = nn.Conv2d(conv_reduce_in, base_channels, kernel_size=1, padding=0, bias=True)
        if with_residual:
            self.conv_residual = ResidualBlock(base_channels, base_channels * 2, base_channels, kernel_size=3, activation1='silu', activation2=None)
        else:
            self.conv_residual = nn.Identity()

    def forward(self, x):
        unshuffled = self.pixel_unshuffle(x)

        if self.preprocessing:
            unshuffled_expanded = self.conv_expand(unshuffled)
            unshuffled_expanded = F.relu(unshuffled_expanded, inplace=False)

            avg_p = self.avg_pool(x)
            max_p = self.max_pool(x)
            min_p_neg = self.max_pool(-x) 
            contrast = max_p + min_p_neg

            conv_2x2 = self.conv_2x2_path(x)
            conv_2x2 = F.relu(conv_2x2, inplace=False)

            concat = torch.cat([unshuffled, unshuffled_expanded, avg_p, max_p, contrast, conv_2x2], dim=1)
        else:
            # Per instructions: unshuffled output goes directly into kPathRB
            concat = unshuffled

        stacked = self.conv_stack(concat)

        # Spatial dimensions now match: both are H/2, W/2
        se_in = torch.cat([unshuffled, stacked], dim=1)

        result = self.se(se_in)
        result = self.conv_reduce(result)
        result = self.conv_proj(unshuffled) + result # Global skip connection from unshuffled input
        result = self.conv_residual(result)

        return result, unshuffled

class RefineWithUnshuffle(nn.Module):
    """Refinement that projects the head pixel-unshuffled features with a
    1x1 conv to `proj_ch`, adds them to the ResidualBlock output (element-wise),
    then applies PixelShuffle directly. The ResidualBlock must output
    `proj_ch` channels so the addition is valid.
    """
    def __init__(self, res_in_ch, res_mid_ch, proj_ch, unshuffled_ch, lores_only=False):
        super().__init__()
        # Residual block produces `proj_ch` channels so it can be added.
        self.res = ResidualBlock(res_in_ch, res_mid_ch, proj_ch, kernel_size=3, activation1='prelu', activation2=None, activation1_params={'num_parameters': res_mid_ch}, inplace=False)
        # Project unshuffled (e.g., 12) -> proj_ch (e.g., 48) using 1x1 conv
        self.unsh_proj = nn.Conv2d(unshuffled_ch, proj_ch, kernel_size=1, stride=1, bias=False)
        # Direct pixel shuffle; no additional conv required per request
        self.ps = nn.PixelShuffle(4 if lores_only else 2)

    def forward(self, x, unshuffled):
        x = self.res(x)
        unsh_p = self.unsh_proj(unshuffled)
        x = x + unsh_p
        x = self.ps(x)
        return x

class ResidualUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 output_channels=3,
                 base_channels=48,
                 max_channels=256,
                 unet_depth=3,
                 blocks_per_level_encoder=2,
                 blocks_per_level_decoder=2,
                 k_paths_head=3,
                 internal_block_channels_ratio=1.0,
                 lores_only=False,
                 verbose=False,
                 d_state=16):
        super().__init__()
        self.verbose = verbose
        self.unet_depth = unet_depth
        self.output_channels = output_channels
        self.lores_only = lores_only
        self.base_channels = base_channels
        # Number of channels produced by HeadProcessing.pixel_unshuffle
        self.unshuffled_ch = input_channels * 4

        # Add a 1x1 convolution to break MiGraphX's layout propagation
        self.skip_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self.perceptual_criterion = PerceptualLoss(
            pixel_loss_type='charbonnier',
            pixel_loss_weight=1.00,
            vgg_weight=0.01,
            high_frequency_type='prewitt',
            high_frequency_weight=0.08,
            lambda_lum=0.0,
            input_is_linear=False
        )

        # --- Head ---
        self.head = HeadProcessing(input_channels, base_channels, k_paths_head, onebyone_expansion=3.0, twobytwo_expansion=4.0, preprocessing=not lores_only, with_residual=False)

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        in_ch = base_channels

        # Helper to cap channels
        def get_ch(depth_idx):
            ch = base_channels * (2 ** depth_idx)
            return int(min(ch, max_channels))

        for d in range(unet_depth):
            out_ch = get_ch(d)

            # Account for PixelUnshuffle expansion (x4) for levels d > 0 input logic
            # Logic: Input to Level 0 is base_channels.
            # Input to Level 1 is Level 0 out (base) * 4 (due to unshuffle downsample)
            prev_ch = get_ch(d-1) if d > 0 else base_channels
            block_in_ch = prev_ch * 4 if d > 0 else in_ch

            # Optimized: Use Standard ResidualBlock
            blocks = [ResidualBlock(block_in_ch if i == 0 else out_ch,
                                    int(out_ch * internal_block_channels_ratio),
                                    out_ch, kernel_size=3,
                                    activation1='prelu',
                                    activation1_params={'num_parameters': int(out_ch * internal_block_channels_ratio)},
                                    activation2=None)
                      for i in range(blocks_per_level_encoder)]
            self.encoder_blocks.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.downs = nn.ModuleList([nn.PixelUnshuffle(2) for _ in range(max(0, unet_depth - 1))])

        # --- Bottleneck ---
        if unet_depth > 0:
            bottleneck_ch = get_ch(unet_depth - 1)
            self.bottleneck_ch = bottleneck_ch
            self.bottleneck = CrossScanMambaBottleneck(bottleneck_ch, d_state=d_state)
        else:
            # When UNet depth is zero we skip bottleneck processing and propagate head features
            bottleneck_ch = base_channels
            self.bottleneck = nn.Identity()
            self.bottleneck_ch = bottleneck_ch

        # --- Decoder ---
        self.ups = nn.ModuleList([nn.PixelShuffle(2) for _ in range(max(0, unet_depth - 1))])
        self.decoder_blocks = nn.ModuleList()

        prev_out_ch = bottleneck_ch
        refine_out_channels = output_channels * (16 if lores_only else 4)

        for d in reversed(range(unet_depth)):
            current_level_ch = get_ch(d)

            if d == 0:
                # Final Stage (No Upsampling)
                in_ch = prev_out_ch + base_channels
                out_ch = current_level_ch + base_channels
                blocks = [
                    ResidualBlock(
                        in_ch if i == 0 else out_ch,
                        int(out_ch * internal_block_channels_ratio),
                        out_ch,
                        kernel_size=3,
                        activation1='prelu',
                        activation1_params={'num_parameters': int(out_ch * internal_block_channels_ratio)},
                        activation2=None
                    )
                    for i in range(blocks_per_level_decoder)
                ]
                self.decoder_blocks.append(nn.Sequential(*blocks))
                prev_out_ch = out_ch
            else:
                # Intermediate Stage (Upsampling via PixelShuffle happens before this block)
                # Input is: (Prev // 4) + Skip
                upsampled_ch = prev_out_ch // 4

                skip_ch = get_ch(d - 1)

                in_ch = upsampled_ch + skip_ch
                out_ch = current_level_ch

                blocks = [ResidualBlock(in_ch if i == 0 else out_ch,
                                        int(out_ch * internal_block_channels_ratio),
                                        out_ch, kernel_size=3,
                                        activation1='prelu',
                                        activation1_params={'num_parameters': int(out_ch * internal_block_channels_ratio)},
                                        activation2=None)
                          for i in range(blocks_per_level_decoder)]
                self.decoder_blocks.append(nn.Sequential(*blocks))

                prev_out_ch = out_ch

        # --- Refinement ---
        proj_ch = refine_out_channels
        self.refine = RefineWithUnshuffle(
            res_in_ch=prev_out_ch,
            res_mid_ch=int(base_channels * 2),
            proj_ch=proj_ch,
            unshuffled_ch=self.unshuffled_ch,
            lores_only=lores_only
        )

        if self.lores_only:
            # Non-learnable depthwise convolution that takes the average of each 2x2 block to create a lores version of the input.
            self.towards_lores = nn.Conv2d(3, 3, kernel_size=2, stride=2, bias=False, groups=3)
            with torch.no_grad():
                self.towards_lores.weight.fill_(0.25)
            self.towards_lores.weight.requires_grad = False # Make weights non-learnable

    def forward(self, x):
        x_in = x

        # Slice for lores only mode; we can drop every second pixel in both dimensions
        if self.lores_only:
            # Assume input is an Amiga lores image
            x = self.towards_lores(x)

        x_in_protected = self.skip_conv(x_in)  # Protect the original input for the global skip connection

        # Analyse different resolutions
        x_head, x_unshuffled = self.head(x)

        if self.verbose:
            print(f"[Head] {x_head.shape}")

        encoder_features = []

        # If UNet depth is zero, skip encoder/bottleneck/decoder entirely and use head output
        if self.unet_depth == 0:
            x = x_head
        else:
            # Level 0
            x = self.encoder_blocks[0](x_head)
            encoder_features.append(x)

            # Levels 1 to Depth-1
            for d in range(1, self.unet_depth):
                x = self.downs[d - 1](x)
                x = self.encoder_blocks[d](x)
                encoder_features.append(x)

        # Ensure the feature channels match the bottleneck expectation. If the encoder
        # didn't increase channels (e.g., unusual path), project from head channels.
        if x.shape[1] != self.bottleneck_ch:
            if getattr(self, 'head_to_bottleneck', None) is not None and x.shape[1] == self.base_channels:
                x = self.head_to_bottleneck(x)
            else:
                raise RuntimeError(f"Channel mismatch before bottleneck: got {x.shape[1]}, expected {self.bottleneck_ch}")

        x = self.bottleneck(x)

        # Decoder path
        for i, block in enumerate(self.decoder_blocks):
            d_val = self.unet_depth - 1 - i

            if i < len(self.ups):
                x_up = self.ups[i](x)
            else:
                x_up = x

            skip = x_head if d_val == 0 else encoder_features[d_val - 1]

            # Pad if necessary
            if x_up.shape[2:] != skip.shape[2:]:
                diffY = skip.size(2) - x_up.size(2)
                diffX = skip.size(3) - x_up.size(3)
                x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            x = torch.cat([x_up, skip], dim=1)
            x = block(x)

        x = self.refine(x, x_unshuffled)
        x = x + x_in_protected  # Global skip connection from input to output
        return x

    def criterion(self, output, target):
        return self.perceptual_criterion(output, target)

    def benchmark(self, input_tensor, warmup_iters=20, test_duration=20.0):
        self.eval()
        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = self(input_tensor)

            start_time = time.time()
            iterations = 0
            while time.time() - start_time < test_duration:
                _ = self(input_tensor)
                iterations += 1

        elapsed = time.time() - start_time
        fps = iterations / elapsed
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_size_mb = param_count * 2 / (1024 ** 2)
        return {
            'fps': fps,
            'params': param_count,
            'size_mb': param_size_mb,
            'output_shape': self(input_tensor).shape
        }

def get_model(name: str = 'light', lores_only: bool = False, verbose: bool = False):
    """
    Returns a selected model configuration.
    """
    if name == 'light':
        return ResidualUNet(
            unet_depth=3 if lores_only else 4,
            blocks_per_level_encoder=1,
            blocks_per_level_decoder=2,
            base_channels=36,
            max_channels=128,
            k_paths_head=4,
            internal_block_channels_ratio=2.0 if lores_only else 1.5,
            lores_only=lores_only,
            verbose=verbose,
            d_state=16 if lores_only else 32
        )
    elif name == 'heavy':
        return ResidualUNet(
            unet_depth=3 if lores_only else 4,
            blocks_per_level_encoder=2 if lores_only else 3,
            blocks_per_level_decoder=4 if lores_only else 4,
            base_channels=96 if lores_only else 38,
            max_channels=256,
            k_paths_head=8 if lores_only else 6,
            internal_block_channels_ratio=2.0 if lores_only else 1.5,
            lores_only=lores_only,
            verbose=verbose,
            d_state=16 if lores_only else 32
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test performance')
    parser.add_argument('--model_type', type=str, required=True, choices=['light', 'heavy'], help='Type of model: light, heavy')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for benchmarking')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile for debugging.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose printing for debugging.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the model state_dict.')
    parser.add_argument('--lores_only', action='store_true', help='Use lores only mode.')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model_type, lores_only=args.lores_only, verbose=args.verbose).to(device)

    if args.save_model:
        print(f"Saving model state_dict to {args.save_model}")
        torch.save(model.state_dict(), args.save_model)
        print("Model saved successfully.")

    if args.verbose:
        print(f"Model: {model} ")
    model = model.half().eval()

    print("Attempting to compile model...")
    if not args.no_compile:
        try:
            model = torch.compile(model, mode="default", fullgraph=True)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Falling back to eager mode.")
            model = model.to(device)
    else:
        print("torch.compile disabled for debugging.")
        model = model.to(device)

    dummy_input = torch.rand((args.batch_size, 3, 576, 736), dtype=torch.float16).to(device)
    results = model.benchmark(dummy_input)

    print("\n--- Results ---")
    print(f"Model output shape: {results['output_shape']}")
    print(f"Model size (trainable parameters): {results['params']}")
    print(f"Model size (MB, assuming float16): {results['size_mb']:.2f} MB")
    print(f"Average FPS: {results['fps']:.2f}")
    print("---------------")