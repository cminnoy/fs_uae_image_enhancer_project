def srgb_to_linear(t):    
    """
    Perfect function.
    t must be in range 0..1
    """
    linear = torch.where(t <= 0.04045,
                         t / 12.92,
                         ((t + 0.055) / 1.055) ** 2.4)
    return linear

def srgb_to_linear_approx(t):
    """Fast approximation of sRGB to linear (gamma ≈ 2.2)."""
    return t ** 2.2

def srgb_to_linear_poly(t):
    """Polynomial approximation to sRGB to linear"""
    return 0.012522878 * t**3 + 0.682171111 * t**2 + 0.305306011 * t

def linear_to_srgb(t):
    """
    Perfect function.
    t must be in range 0..1
    """
    srgb = torch.where(t <= 0.0031308,
                       t * 12.92,
                       1.055 * (t ** (1.0 / 2.4)) - 0.055)
    return srgb

def linear_to_srgb_approx(t):
    """Fast approximation of linear to sRGB (gamma ≈ 1/2.2)."""
    return t ** (1.0 / 2.2)

def linear_to_srgb_poly(t):
    """Polynomial approximation to linear to sRGB"""
    return 0.585122381 * t**3 - 0.164759123 * t**2 + 0.579636742 * t