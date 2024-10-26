
import numpy as np
import scipy.constants

# Gaussian process regression
import sklearn.gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, RBF

def add_noise_to_layers(layers, snr_db, domain_x,
                        prf = 10, velocity = 50, center_frequency = 60e6,
                        altitude = 500, dielectric_permittivity=3.17,
                        cross_track_slope_angle_deg=0,
                        rng=np.random.default_rng()):
    """
    Adds noise to the layer positions. Noise is simulated based on the radar
    system and platform parameters. New layer interpolation functions are returned
    with the noise added.

    Parameters:
    - layers (list): List of layer functions representing the layer positions.
    - snr_db (float): Signal-to-noise ratio in decibels.
    - domain_x (float): Length of the domain in the x-direction.
    - prf (float, optional): Pulse repetition frequency in Hz
    - velocity (float, optional): Velocity of the radar system in m/s
    - center_frequency (float, optional): Center frequency of the radar system in Hz
    - altitude (float, optional): Altitude of the radar system in m
    - dielectric_permittivity (float, optional): Dielectric permittivity of the medium. Defaults to 3.17.
    - cross_track_slope_angle_deg (float, optional): Cross-track slope angle in degrees
    - rng (numpy.random.Generator, optional): Random number generator. Defaults to np.random.default_rng().
    Returns:
    - layers_measured (list): List of layer functions representing the measured layer positions with added noise.
    """

    # Convert SNR to linear units
    snr_linear = 10**(snr_db/10) # diml
    # Convert angle to radians
    cross_track_slope_angle = np.deg2rad(cross_track_slope_angle_deg)

    pulse_spacing = velocity / prf # m
    pulses_x = np.arange(0, domain_x, pulse_spacing)

    ph_noise = rng.normal(0, np.sqrt(1/snr_linear), (len(pulses_x), len(layers))) # shape: (pulses, layers)
    layer_position_noise = scipy.constants.c * ph_noise / (center_frequency * np.sqrt(dielectric_permittivity) * 4 * np.pi)

    # Error due to off-nadir reflections
    layer_depths = np.array([layers[idx](pulses_x) for idx in range(len(layers))])
    layer_depths = np.swapaxes(layer_depths, 0, 1)
    layer_depths += altitude
    err_r_off_nadir = layer_depths * (1 - np.cos(cross_track_slope_angle))
    layer_position_noise -= err_r_off_nadir

    print(layer_position_noise.shape)

    layers_measured = []
    for idx in range(len(layers)):
        layer_with_noise_interp = scipy.interpolate.PchipInterpolator(pulses_x, layer_position_noise[:, idx] + layers[idx](pulses_x))
        layers_measured.append(layer_with_noise_interp)
    
    return layers_measured



# =============================================================================

def simulate_stacking(layers, domain_x, kernel_length_m=100, velocity=50, prf=10):

    pulse_spacing = velocity / prf # m
    pulses_x = np.arange(0, domain_x, pulse_spacing)

    kernel_size = int(kernel_length_m / pulse_spacing)
    if kernel_size % 2 == 0:
        kernel_size -= 1

    kernel = np.ones(kernel_size)
    kernel /= np.sum(kernel)

    if len(kernel) % 2 == 0:
        raise ValueError("Kernel must have an odd number of elements")

    def smooth_layer(layer_values):
        padded = np.zeros(len(layer_values) + len(kernel) - 1)
        padded[(len(kernel)-1)//2:-((len(kernel)-1)//2)] = layer_values
        padded[:len(kernel)//2] = layer_values[0]
        padded[-len(kernel)//2:] = layer_values[-1]
        return np.convolve(padded, kernel, mode='valid')

    layers_movmean = []
    
    for layer in layers:
        layer_values_smoothed = smooth_layer(layer(pulses_x))
        layers_movmean.append(scipy.interpolate.PchipInterpolator(pulses_x, layer_values_smoothed))
    
    return layers_movmean

def gp_smoothing(layers, domain_x, x_spacing,
                 kernel = RBF(length_scale=1e3, length_scale_bounds=(1, 50e3)) + WhiteKernel(1e-5, noise_level_bounds=(1e-10, 1)),
                 initialize_with_prior_kernel=True):
    
    xs_tmp = np.arange(0, domain_x, x_spacing)

    def gp_interpolate_layer(layer_x, layer_z, kernel):
        gaussian_process = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gaussian_process.fit(layer_x.reshape(-1, 1), layer_z)
        # def interp(x):
        #     if np.isscalar(x):
        #         x = np.array([x])
        #     return gaussian_process.predict(x.reshape(-1, 1))

        interp = scipy.interpolate.PchipInterpolator(xs_tmp, gaussian_process.predict(xs_tmp.reshape(-1, 1)))
        
        return interp, gaussian_process

    layers_smoothed = []

    

    for idx, layer in enumerate(layers):
        layer_values = layer(xs_tmp)
        interp, gp = gp_interpolate_layer(xs_tmp, layer_values, kernel)
        layers_smoothed.append(interp)
        if initialize_with_prior_kernel:
            kernel = gp.kernel_ # Use the kernel from the previous layer to initialize the next layer
        print(f"[Layer {idx}] {gp.kernel_}")

    return layers_smoothed

# ====

# if apply_smoothing:
#     # Step 1
#     # Apply smoothing

#     kernel_length_m = 100 # meters
#     kernel_size = int(kernel_length_m / pulse_spacing)
#     if kernel_size % 2 == 0:
#         kernel_size -= 1

#     kernel = np.ones(kernel_size)
#     kernel /= np.sum(kernel)

#     if len(kernel) % 2 == 0:
#         raise ValueError("Kernel must have an odd number of elements")

#     def smooth_layer(layer_values):
#         padded = np.zeros(len(layer_values) + len(kernel) - 1)
#         padded[(len(kernel)-1)//2:-((len(kernel)-1)//2)] = layer_values
#         padded[:len(kernel)//2] = layer_values[0]
#         padded[-len(kernel)//2:] = layer_values[-1]
#         return np.convolve(padded, kernel, mode='valid')

#     layers_t0_movmean = []
#     layers_t1_movmean = []
#     for layer in layers_t0_measured:
#         layer_values_smoothed = smooth_layer(layer(pulses_x))
#         layers_t0_movmean.append(scipy.interpolate.PchipInterpolator(pulses_x, layer_values_smoothed))
        
#     for layer in layers_t1_measured:
#         layer_values_smoothed = smooth_layer(layer(pulses_x))
#         layers_t1_movmean.append(scipy.interpolate.PchipInterpolator(pulses_x, layer_values_smoothed))

#     # Step 2
#     # Apply smoothing via gaussian process regression

#     def gp_interpolate_layer(layer_x, layer_z, kernel):
#         gaussian_process = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y=True)
#         gaussian_process.fit(layer_x.reshape(-1, 1), layer_z)
#         def interp(x):
#             if np.isscalar(x):
#                 x = np.array([x])
#             return gaussian_process.predict(x.reshape(-1, 1))
#         return interp, gaussian_process

#     kernel = RBF(length_scale=1e3, length_scale_bounds=(1, 50e3)) + WhiteKernel(1e-5, noise_level_bounds=(1e-10, 1))
#     layers_t0_smoothed = []
#     layers_t1_smoothed = []

#     xs_tmp = np.arange(0, domain_x, kernel_length_m)

#     for idx, layer in enumerate(layers_t0_movmean):
#         layer_values = layer(xs_tmp)
#         interp, gp = gp_interpolate_layer(xs_tmp, layer_values, kernel)
#         layers_t0_smoothed.append(interp)
#         kernel = gp.kernel_ # Use the kernel from the previous layer to initialize the next layer
#         print(f"[t0, layer {idx}] {gp.kernel_}")
        
#     for idx, layer in enumerate(layers_t1_movmean):
#         layer_values = layer(xs_tmp)
#         interp, gp = gp_interpolate_layer(xs_tmp, layer_values, kernel)
#         layers_t1_smoothed.append(interp)
#         kernel = gp.kernel_ # Use the kernel from the previous layer to initialize the next layer
#         print(f"[t1, layer {idx}] {gp.kernel_}")

# else:
#     layers_t0_smoothed = layers_t0_measured
#     layers_t1_smoothed = layers_t1_measured