def create_layer_finite_difference_fns(layers_t0_measured, layers_t1_measured):
    """
    Creates finite difference functions for numerical approximations of layer geometry
    
    Parameters:
    layers_t0_measured (list): List of functions representing the measured layer properties at time t0.
    layers_t1_measured (list): List of functions representing the measured layer properties at time t1.
    
    Returns:
    tuple: A tuple containing the following finite difference functions:
        - layer_dl_dx: Numerical central difference approximation of layer slope.
        - layer_dl_dt: Numerical forward difference approximation of layer vertical deformation.
        - layer_d2l_dxdz: Numerical central difference approximation of d^2l/(dxdz).
        - layer_d2l_dtdz: Numerical central difference approximation of d^2l/(dtdz).

    Note:
    - The output units for layer_dl_dx are m/m.
    - The output units for layer_dl_dt are m/year.
    - The output units for layer_d2l_dxdz are m/m^2.
    - The output units for layer_d2l_dtdz are m/(m*year).
    """

    def layer_dl_dx(x, layer_idx, dx=1):
        """
        Numerical central difference approximation of layer slope
        Output units are m/m
        """
        return (layers_t0_measured[layer_idx](x+(dx/2)) - layers_t0_measured[layer_idx](x-(dx/2))) / dx

    def layer_dl_dt(x, layer_idx):
        """
        Numerical forward difference approximation of layer vertical deformation
        Output units are m/year
        """
        return (layers_t1_measured[layer_idx](x) - layers_t0_measured[layer_idx](x))

    def layer_d2l_dxdz(x, layer_idx):
        """
        Numerical central difference approximation of d^2l/(dxdz)
        Output units are m/m^2
        """
        layer_p1 = layer_dl_dx(x, layer_idx+1)
        layer_m1 = layer_dl_dx(x, layer_idx-1)
        layer_dz = layers_t0_measured[layer_idx+1](x) - layers_t0_measured[layer_idx-1](x)
        return (layer_p1 - layer_m1) / (layer_dz)

    def layer_d2l_dtdz(x, layer_idx):
        """
        Numerical central difference approximtion of d^2l/(dtdz)
        Output units are m/(m*year)
        """
        layer_p1 = layer_dl_dt(x, layer_idx+1)
        layer_m1 = layer_dl_dt(x, layer_idx-1)
        layer_dz = layers_t0_measured[layer_idx+1](x) - layers_t0_measured[layer_idx-1](x)
        return (layer_p1 - layer_m1) / (layer_dz)
    
    return layer_dl_dx, layer_dl_dt, layer_d2l_dxdz, layer_d2l_dtdz