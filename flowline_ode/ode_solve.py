import numpy as np
import scipy.constants
import scipy.integrate
from tqdm import tqdm


def solve_layer_ode(d2l_dxdz_fn, d2l_dtdz_fn, bounds, initial_velocity, solve_args):

    def du_dtau(tau, u):
        return -1 * d2l_dxdz_fn(tau)*u - d2l_dtdz_fn(tau)
    
    return scipy.integrate.solve_ivp(du_dtau, bounds, np.array([initial_velocity]), dense_output=True, **solve_args)


def solve_all_layers(layers_t0, layer_d2l_dxdz, layer_d2l_dtdz, x, z, domain_x, xs_layers, u = None, start_pos_x = 0, solve_args={}):

    # Default solve_args
    if 'max_step' not in solve_args:
        solve_args['max_step'] = 100

    layer_solutions = {}

    for idx in tqdm(np.arange(1, len(layers_t0)-1)):
        layer = layers_t0[idx]
        if u is not None:
            u0 = u.subs([(x, start_pos_x), (z, layer(start_pos_x))]).evalf() * scipy.constants.year
        else:
            u0 = 0.0 # Assume zero starting velocity -- generally fine if it's close to being correct
        # If you're extracting points from the solution (for rheology estimates, for example), set max_step to no more than the spacing at which you'll extract points
        # (but if you're just plotting velocity, this will run a lot faster if you let the solver pick a max step size)
        in_lower_guardrails = xs_layers[layer(xs_layers) < 200]
        if len(in_lower_guardrails) > 0:
            layer_end = np.min(in_lower_guardrails)
        else:
            layer_end = domain_x
        
        # Functions for layer
        d2l_dxdz_fn = lambda tau: layer_d2l_dxdz(tau, idx)
        d2l_dtdz_fn = lambda tau: layer_d2l_dtdz(tau, idx)
        # Solve layer ODE
        layer_solutions[idx] = solve_layer_ode(d2l_dxdz_fn, d2l_dtdz_fn, [start_pos_x, layer_end], u0, solve_args)
        
        if layer_solutions[idx].status != 0:
            print(f"Layer {idx} failed to solve")
            print(layer_solutions[idx].message)
        
    return layer_solutions