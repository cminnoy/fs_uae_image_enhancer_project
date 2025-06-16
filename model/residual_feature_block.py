import torch.nn as nn
import copy
import activations

class ResidualFeatureBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, acts={
        'act1': 'identity',
        'act1_params': None,
        'act2': 'relu',
        'act2_params': None,
        'act3': 'identity',
        'act3_params': None,
        'act4': 'relu',
        'act4_params' : None
    }):
        super(ResidualFeatureBlock, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # Handle num_parameters for all activations
        acts = copy.deepcopy(acts)
        for act_key, ch in zip(
            ['act1', 'act2', 'act3', 'act4'],
            [mid_channels, mid_channels, out_channels, out_channels]
        ):
            params_key = f"{act_key}_params"
            if isinstance(acts.get(params_key), dict):
                num_params = acts[params_key].get('num_parameters')
                if num_params == 'global':
                    acts[params_key]['num_parameters'] = 1
                elif num_params == 'channel':
                    acts[params_key]['num_parameters'] = ch
        self.act1 = activations.get_activation(acts['act1'], params=acts['act1_params'])
        self.act2 = activations.get_activation(acts['act2'], params=acts['act2_params'])
        self.act3 = activations.get_activation(acts['act3'], params=acts['act3_params'])
        self.act4 = activations.get_activation(acts['act4'], params=acts['act4_params'])
        self.proj_conv = None
        if in_channels != out_channels:
            self.proj_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        if self.proj_conv:
            identity = self.proj_conv(identity)
        x = identity + x
        x = self.act4(x)
        return x