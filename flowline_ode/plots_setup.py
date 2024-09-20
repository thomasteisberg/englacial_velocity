import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants

def make_below_surface_mask(X, Z, surface=None):
    below_surface_mask = np.ones_like(X, dtype=float)
    if surface:
        below_surface = (Z < surface(X))
        below_surface_mask[~below_surface] = np.nan
    return below_surface_mask

def plot_surface(xs, surface, ds_dx=None, subplots_params={}):
    """
    Plot the surface and its derivative with twin x axes.

    Parameters:
    xs (array-like): The x-coordinates of the surface points to plot.
    surface (callable): A function that takes xs as input and returns the surface elevation.
    ds_dx (callable, optional): A function that takes xs as input and returns the derivative of the surface elevation.
    subplots_params (dict, optional): Additional parameters to pass to the `subplots` function.

    Returns:
    tuple: A tuple containing the figure and the axis or axes object(s) created.

    If `ds_dx` is provided, the tuple will also contain the right y-axis axes object.

    """
    fig, ax = plt.subplots(figsize=(8, 4), **subplots_params)
    
    # Surface elevation
    line_surf = ax.plot(xs/1e3, surface(xs), 'blue', label='s(x) [m]')
    ax.tick_params(axis='y', colors='blue')
    ax.set_ylabel('Surface elevation [m]', color='blue')
    lns = line_surf # Store for legend making
    
    # Surface slope
    if ds_dx:
        ax_right = ax.twinx()
        ax_right.tick_params(axis='y', colors='red')
        line_slope = ax_right.plot(xs/1e3, (180/np.pi) * np.tan(ds_dx(xs)), 'r--', label='ds_dx(s) [deg]')
        ax_right.set_ylabel('Surface slope [deg]', color='red')
        lns = line_surf + line_slope

    # Create legend (spanning both axes if using a right axis)
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right')

    # Title and axis labels
    ax.set_title('Surface and surface slope')
    ax.set_xlabel('Distance [km]')
    ax.grid(True, axis='x')

    if ds_dx:
        return fig, (ax, ax_right)
    else:
        return fig, ax
    

def plot_velocity(xs, zs, x, z, u, w, surface=None, domain_z=None):
    """
    Plot the horizontal and vertical velocity fields.

    Parameters:
    xs (array-like): Array of x-coordinates.
    zs (array-like): Array of z-coordinates.
    x (symbol): Symbolic x-coordinate variable.
    z (symbol): Symbolic z-coordinate variable.
    u (symbol): Symbolic expression for horizontal velocity.
    w (symbol): Symbolic expression for vertical velocity.
    surface (function, optional): Function defining the surface elevation as a function of x. Defaults to None.
    domain_z (float, optional): Maximum value of z-coordinate. Defaults to None.

    Returns:
    fig: The generated figure.
    (ax_U, ax_W) (tuple): Tuple containing the two subplots for horizontal and vertical velocity.

    """
    X, Z = np.meshgrid(xs, zs)
    U = sympy.lambdify((x, z), u, modules='numpy')(X, Z)
    W = sympy.lambdify((x, z), w, modules='numpy')(X, Z)

    
    below_surface_mask = make_below_surface_mask(X, Z, surface)

    fig, (ax_U, ax_W) = plt.subplots(2,1, figsize=(8, 6), sharex=True)
    pcm_U = ax_U.pcolormesh(X/1e3, Z, scipy.constants.year*U*below_surface_mask, cmap='viridis', vmin=0)
    fig.colorbar(pcm_U, ax=ax_U, label='Horizontal velocity [m/yr]')
    ax_U.set_title('Horizontal velocity')
    ax_U.set_ylabel('z [m]')
    ax_U.set_ylim(0, domain_z)

    pcm_W = ax_W.pcolormesh(X/1e3, Z, scipy.constants.year*W*below_surface_mask, cmap='viridis')
    fig.colorbar(pcm_W, ax=ax_W, label='Vertical velocity [m/yr]')
    ax_W.set_title('Vertical velocity')
    ax_W.set_xlabel('x [km]')
    ax_W.set_ylabel('z [m]')
    fig.tight_layout()

    return fig, (ax_U, ax_W)


def plot_surface_bed_velocity(xs, x, z, u, surface_sym):
    """
    Plot the surface and basal velocities of a flowline.

    Parameters:
    xs (numpy.ndarray): Array of x-coordinates.
    x (sympy.Symbol): Symbol representing the x-coordinate.
    z (sympy.Symbol): Symbol representing the vertical coordinate.
    u (sympy.Expr): Expression representing the velocity field.
    surface_sym (sympy.Symbol): Symbol representing the surface elevation.

    Returns:
    matplotlib.figure.Figure: The generated figure.
    matplotlib.axes.Axes: The generated axes.

    """
    # Calculate the surface velocity (just below the surface to avoid border effects)
    u_surface = sympy.lambdify(x, u.subs(z, surface_sym-0.1), modules='numpy')(xs)
    u_bed = sympy.lambdify(x, u.subs(z, 0.1), modules='numpy')(xs)

    # Plot the surface velocity
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs/1e3, scipy.constants.year*u_surface, 'k-', label='Surface velocity [m/yr]')
    ax.plot(xs/1e3, scipy.constants.year*u_bed, 'r-', label='Basal velocity [m/yr]')
    #ax.set_title('Surface velocity')
    ax.legend()
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('Horizontal Velocity [m/yr]')
    ax.grid(True)

    return fig, ax

def plot_velocity_magnitude(xs, zs, x, z, u, w, surface=None, domain_x=None, layers=None, xs_layers=None, title="Velocity magnitude"):
    # Plot the advected layers on top of a pcolormesh plot of the velocity magnitude
    X, Z = np.meshgrid(xs, zs)
    U = sympy.lambdify((x, z), u, modules='numpy')(X, Z)
    W = sympy.lambdify((x, z), w, modules='numpy')(X, Z)
    vel = np.sqrt(U**2 + W**2)

    fig, ax = plt.subplots(figsize=(8, 4))
    pcm = ax.pcolormesh(X/1e3, Z, scipy.constants.year * vel * make_below_surface_mask(X, Z, surface), cmap='viridis', alpha=0.8) #vmax=170)
    fig.colorbar(pcm, ax=ax, label='Velocity magnitude [m/yr]')
    
    if layers:
        if xs_layers is None:
            xs_layers = xs
        for layer in layers:
            ax.plot(xs_layers/1e3, layer(xs_layers), 'k--')
    
    ax.set_title(title)
    ax.grid(True)
    if domain_x:
        ax.set_xlim(0, domain_x/1e3)
    if surface:
        ax.set_ylim(0, surface(0))
    ax.set_xlabel('x [km]')
    ax.set_ylabel('z [m]')

    return fig, ax