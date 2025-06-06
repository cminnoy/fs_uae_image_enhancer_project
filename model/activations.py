import torch
import torch.nn as nn

# --- Custom Activation Modules ---

class TeLU(nn.Module):
    """
    Custom activation function: x * tanh(exp(x)).
    Reference: https://arxiv.org/abs/1812.06579
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.exp(x))

class ScaledTanh(nn.Module):
    """
    Maps the output of Tanh (range [-1, 1]) to the range [0, 1].
    Formula: (tanh(x) + 1) / 2
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(x) + 1.0) * 0.5

class SinLU(nn.Module):
    """
    https://www.mdpi.com/2227-7390/10/3/337
    https://github.com/ashis0013/SinLU
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(1), requires_grad=True)
    def forward(self,x):
        return torch.sigmoid(x)*(x+self.a*torch.sin(self.b*x))
    
class BiasedReLU(nn.Module):
    """
    ReLU with a learnable bias.
    """
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
    def forward(self,x):
        return torch.relu(x - self.bias)

# --- Activation Function Registry ---

ACTIVATION_REGISTRY = {
    # Standard PyTorch activations
    'identity': nn.Identity,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'leaky_relu': nn.LeakyReLU,
    'mish': nn.Mish,
    'prelu': nn.PReLU,
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'sigmoid': nn.Sigmoid,
    'silu': nn.SiLU, # PyTorch's built-in Swish
    'swish': nn.SiLU, # Alias for SiLU
    'softplus': nn.Softplus,
    'tanh': nn.Tanh,

    # Standard PyTorch activations often needing 'dim' parameter
    'log_softmax': nn.LogSoftmax,
    'softmax': nn.Softmax,

    # Custom activations 
    'scaled_tanh': ScaledTanh,
    'telu': TeLU,
    'sinlu': SinLU,
    'biased_relu': BiasedReLU
}

# --- Activation Factory Function ---

def get_activation(activation_name: str, params: dict = None, inplace: bool = False):
    """
    Factory function to create and return an activation module instance by name.

    Args:
        activation_name: The name of the activation function (case-insensitive).
                         Should match keys in ACTIVATION_REGISTRY.
        params: An optional dictionary of parameters to pass to the activation
                module's constructor (e.g., {'negative_slope': 0.05} for LeakyReLU,
                {'num_parameters': 128} for PReLU, {'dim': 1} for Softmax).
                If None, default constructor parameters are used.
        inplace: If True, attempts to create the activation module with inplace=True,
                 if the module supports it. Use with caution.

    Returns:
        An instance of the specified activation module.

    Raises:
        ValueError: If the activation_name is not supported.
        TypeError: If params are provided but the activation module's constructor
                   does not accept them or parameter names are incorrect.
    """
    activation_name_lower = activation_name.lower()

    if activation_name_lower not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported activation: '{activation_name}'. "
            f"Supported activations are: {list(ACTIVATION_REGISTRY.keys())}"
        )

    activation_class = ACTIVATION_REGISTRY[activation_name_lower]

    # Prepare parameters for the constructor
    constructor_params = {}
    if params is not None:
        constructor_params.update(params) # Start with provided params

    # Add inplace=True if the activation class supports it
    # Common activations that support inplace: ReLU, LeakyReLU, ELU, ReLU6, SiLU, Mish
    if inplace:
         constructor_params['inplace'] = True

    try:
        if params is None:
            if activation_class == nn.PReLU:
                 return activation_class()
            elif activation_class in [nn.Softmax, nn.LogSoftmax]:
                 return activation_class(dim=1)
            elif activation_class == nn.LeakyReLU:
                 return activation_class()
            elif activation_class == nn.ELU:
                 return activation_class()
            else:
                 # Ensure inplace is handled if relevant and not in params
                 if inplace and 'inplace' not in constructor_params and hasattr(activation_class, 'inplace'):
                      return activation_class(inplace=True)
                 return activation_class() # No params needed

        else:
             if inplace and 'inplace' not in constructor_params and hasattr(activation_class, 'inplace') and 'inplace' not in activation_class.__init__.__code__.co_varnames:
                  pass

             return activation_class(**constructor_params)


    except TypeError as e:
        raise TypeError(
            f"Error instantiating activation '{activation_name}' with parameters {constructor_params}. "
            f"Check if the parameters match the constructor arguments of {activation_class.__name__}. "
            f"Original error: {e}"
        )

# --- Example Usage ---
# relu_inst = get_activation('relu')
# leaky_relu_inst = get_activation('leaky_relu', params={'negative_slope': 0.05})
# leaky_relu_default = get_activation('leaky_relu') # Uses default negative_slope=0.01
# prelu_inst_shared = get_activation('prelu') # num_parameters=1 by default
# prelu_inst_per_channel = get_activation('prelu', params={'num_parameters': 128, 'init': 0.3})
# swish_inst = get_activation('swish') # This is nn.SiLU
# gelu_inst = get_activation('gelu')
# softmax_inst_dim1 = get_activation('softmax') # Uses default dim=1 in the factory
# softmax_inst_dim_other = get_activation('softmax', params={'dim': 0})
# scaled_tanh_inst = get_activation('scaled_tanh')
# relu_inplace = get_activation('relu', inplace=True)
# leaky_relu_inplace = get_activation('leaky_relu', params={'negative_slope': 0.02}, inplace=True)