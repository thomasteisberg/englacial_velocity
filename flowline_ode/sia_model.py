import numpy as np
import sympy
import scipy.constants
import scipy.integrate
import datetime

def sia_model(x_sym, z_sym, surface_sym, dsdx_sym,
              rho=918, g=9.8, A0=3.985e-13, n_A0=3, Q=60e3, R=8.314,
              T_rel_p=(273.15-20), log_ref_stress=6,
              n=3, basal_velocity_sym=0, enhancement_factor=1):
    """
    Calculates the horizontal and vertical velocities of ice flow using the
    Shallow Ice Approximation (SIA) model.

    Implementation is based on Section 3.4 of Ralf Greve's course notes:
    https://ocw.hokudai.ac.jp/wp-content/uploads/2016/02/DynamicsOfIce-2005-Note-all.pdf

    Our SIA model is fully specified by a surface contour (including its derivative) and a
    basal velocity field (plus some constants).

    Parameters:
    - x_sym: Symbolic variable representing the horizontal coordinate.
    - z_sym: Symbolic variable representing the vertical coordinate.
    - surface_sym: Symbolic variable representing the ice surface elevation.
    - dsdx_sym: Symbolic variable representing the slope of the ice surface.
    - rho: Density of ice in kg/m^3 (default: 918).
    - g: Acceleration due to gravity in m/s^2 (default: 9.8).
    - A0: Flow rate factor in s^-1 Pa^-3 (default: 3.985e-13).
    - n_A0: Flow law exponent for A0 (default: 3).
    - Q: Activation energy in J/mol (default: 60e3).
    - R: Universal gas constant in J/(mol K) (default: 8.314).
    - T_rel_p: Reference temperature in Kelvin (default: 273.15 - 20).
    - log_ref_stress: Reference stress in logarithmic scale (default: 6).
    - n: Flow law exponent (default: 3).
    - basal_velocity_sym: Symbolic variable representing the basal velocity (default: 0).

    Returns:
    - u_sym: Symbolic expression for the horizontal velocity.
    - w_sym: Symbolic expression for the vertical velocity.
    - du_dx_sym: Symbolic expression for the derivative of horizontal velocity with respect to x.
    """

    # Baseline case A value
    A = A0 * np.exp(-Q / (R * T_rel_p))

    if n != n_A0:
        log_ref_strain_rate = np.log10(2*A) + n_A0 * log_ref_stress
        log_2A = log_ref_strain_rate - n * log_ref_stress # log10(2*A)
        A = 10**(log_2A) / 2

    # Solve for u (horizontal velocity) [Eq. 3.92]
    
    # Version with assumed constant A
    # u_sym = (-2.0 * A * sympy.Abs(dsdx_sym)**(n-1.0) * dsdx_sym * rho**n * g**n *
    #          (surface_sym**(n+1.0) - (surface_sym - z_sym)**(n+1.0)) / (n + 1.0)) + basal_velocity_sym

    # Version with spatially-variable A
    # u_sym = basal_velocity_sym - (2.0 * (rho * g)**n *
    #                               sympy.Abs(dsdx_sym)**(n-1.0) * dsdx_sym * 
    #                               sympy.integrate(A *
    #                                               (surface_sym - z_sym)**n,
    #                                               (z_sym, 0, z_sym)) )


    # if n % 1 != 0:
    #     raise ValueError("Flow law exponent n must be an integer")
    # n = int(n)

    u_sym = basal_velocity_sym - (2 * (rho * g)**n *
                                  sympy.Abs(dsdx_sym)**(n-1) * dsdx_sym * 
                                  sympy.integrate(enhancement_factor * A *
                                                  (surface_sym - z_sym)**n,
                                                  (z_sym, 0, z_sym)) )
    
    # Recover w (vertical velocity) from u through incompressibility
    du_dx_sym = sympy.diff(u_sym, x_sym)
    dw_dz_sym = -1 * du_dx_sym

    # Integrate up from the bed to find w(x, z)
    # Assume: w(x, z=0) =0 (no basal melt)
    w_sym = sympy.integrate(dw_dz_sym, (z_sym, 0, z_sym))

    return u_sym, w_sym, du_dx_sym



def lambdify_and_vectorize_if_needed(vars, symbolic_expr, warn_if_vectorizing = True, **kwargs):
    """
    Acts just like sympy.lambdify() except also calls np.vectorize() on the
    result if the function lambdify() returns cannot accept vector inputs.
    (This happens if the returned expression contains an integral. It is not
    clear to me if this is a bug or the intended behavior. Always applying
    np.vectorize() isn't a good approach because the result is much slower
    than the natively vectorized functions from lambdify.)
    """

    l = sympy.lambdify(vars, symbolic_expr, **kwargs)

    # Figure out how many variables the lambdified function expects
    # vars may be either a single Symbol or a list of Symbols
    n_vars = 1
    try:
        n_vars = len(vars)
    except TypeError:
        n_vars = 1

    test_input = [np.array([1, 1]) for i in range(n_vars)]
    must_vectorize = False
    try:
        test_output = l(*test_input)
        if len(test_output) != len(test_input[0]):
            must_vectorize = True
    except:
        must_vectorize = True

    if must_vectorize:
        if warn_if_vectorizing:
            print(f"Warning: np.vectorize() is being called on the lambdified function {symbolic_expr}.")
        return np.vectorize(l)
    
    return l

def uw_on_grid(xs, zs, u, x, z, basal_velocity_fn, domain_z, return_du_dx_fd=False, verbose=False):
    dx = xs[1] - xs[0]
    xs_extended = np.concatenate(([xs[0] - dx], xs, [xs[-1] + dx]))
    
    tstamp = datetime.datetime.now()

    du_dz_fn = sympy.lambdify((x, z), sympy.diff(u, z))

    def du_dz_ode(z, u, x):
        return du_dz_fn(x, z)
    
    sol = scipy.integrate.solve_ivp(du_dz_ode, (0, domain_z), basal_velocity_fn(xs_extended), t_eval=zs,
                                    args=(xs_extended,),
                                    atol=1e-4/scipy.constants.year, rtol=1e-4/scipy.constants.year)
    U_extended = np.transpose(sol.y)
    if verbose:
        print(f"Shape of U_extended: {U_extended.shape}")
        print(f"Time to compute U_extended: {datetime.datetime.now() - tstamp}")
    tstamp = datetime.datetime.now()

    dudx_finite_diff = scipy.signal.correlate(U_extended, np.array([-1, 0, 1])[np.newaxis, :], mode='valid') / (2*(xs[1] - xs[0]))
    W = np.cumsum(-1*dudx_finite_diff, axis=0) * (zs[1] - zs[0])

    U = U_extended[:, 1:-1]

    if verbose:
        print(f"len(xs): {len(xs)}, len(zs): {len(zs)}")
        print(f"Shape of U: {np.shape(U)}")
        print(f"Shape of W: {np.shape(W)}")
        print(f"Time to compute W: {datetime.datetime.now() - tstamp}")

    if return_du_dx_fd:
        return U, W, dudx_finite_diff
    else:
        return U, W


def advect_layer(u, w, xs, initial_layer, layer_ages, xs_initial=None, max_age_timestep=None):
    """
    Advect a layer in a vertical velocity field w and horizontal velocity field u.
    The layer is represented as a 1D array of thicknesses at each x position.

    Advection is done by solving an ODE for sampled particles at locations (xs, initial_layer(xs))
    moving through the velocity field (u, w). The ODE is solved for each layer age.

    Args:
    u: Function mapping (x, z) to horizontal velocity at that position
    w: Function mapping (x, z) to vertical velocity at that position
    xs: 1D array of x positions
    initial_layer: A function mapping x positions to the initial layer position
    layer_ages: 1D array of ages of each layer to be output
    xs_initial: Optional 1D array of x positions to use for the initial layer positions. If None, use xs.

    Returns:
    An array (of the same length as layer_ages) of functions mapping x positions to the layer position at that age
    """

    # Use the same x positions for the initial layer as the output x positions if not specified
    if xs_initial is None:
        xs_initial = xs

    zs_initial = initial_layer(xs_initial)

    # Encode x and y coordinates into flattened list for scipy.integrate.solve_ivp
    y = np.concatenate([xs_initial, zs_initial])

    # Simple 2D advection ODE
    def ode_fun(t, y):
        x = y[:len(y)//2]
        z = y[len(y)//2:]
        us = u(x, z)
        ws = w(x, z)

        dydt = np.concatenate([us, ws])
        return dydt
    
    # Integrate the ODE for each layer age
    # Need to do it this way rather than solving for all ages at once because
    # points move too far out of the domain. Re-interpolating them back onto
    # normal x spacing between time steps fixes this.
    t = 0
    advected_layers = []
    for idx, age in enumerate(layer_ages):
        #print(f"Layer idx {idx}")
        if max_age_timestep is not None:
            while (age-t) > max_age_timestep:
                print(f"Requested next step age: {age}, delta: {age - t}, actually taking step of size {max_age_timestep}")
                sol = scipy.integrate.solve_ivp(ode_fun, [0, max_age_timestep], y, dense_output=True)#, rtol=1e-8, atol=1e-8)
                xs_advected = sol.y[:len(y)//2, -1]
                zs_advected = sol.y[len(y)//2:, -1]

                sorted_indices = np.argsort(xs_advected)
                layer_interp_for_next_layer = scipy.interpolate.PchipInterpolator(xs_advected[sorted_indices], zs_advected[sorted_indices])
                
                y = np.concatenate([xs, layer_interp_for_next_layer(xs)])
                t += max_age_timestep

        sol = scipy.integrate.solve_ivp(ode_fun, [0, age-t], y, dense_output=False, rtol=1, atol=1)#, rtol=1e-8, atol=1e-8)
        xs_advected = sol.y[:len(y)//2, -1]
        zs_advected = sol.y[len(y)//2:, -1]

        sorted_indices = np.argsort(xs_advected)
        layer_interp_for_next_layer = scipy.interpolate.PchipInterpolator(xs_advected[sorted_indices], zs_advected[sorted_indices])
        
        y = np.concatenate([xs, layer_interp_for_next_layer(xs)])
        t = age

        layer_interp = layer_interp_for_next_layer

        advected_layers.append(layer_interp)

    return advected_layers
