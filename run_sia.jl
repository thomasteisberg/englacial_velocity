using Revise

using CairoMakie
using Symbolics
using OrderedCollections

ENV["PYTHON"] = ""
using PyCall
scipy_interpolate = pyimport_conda("scipy.interpolate", "scipy")

includet("sia.jl")
includet("age_depth.jl")
includet("horizontal_velocity.jl")
includet("plot_helpers.jl")

seconds_per_year = 60 * 60 * 24 * 365.0 # m/s to m/yr conversion

#  ============================
## Problem definition and setup
#  ============================

# Domain size
domain_x = 10000.0 # meters
domain_z = 1500.0 # meters

# Grids for when discretization is needed
dx = 100.0
dz = 50.0
xs = 0.0:dx:domain_x
zs = 0.0:dz:domain_z

# x, z are our spatial coordinates
# These will be used in various symbolic math throughout
@parameters x z

# Define surface geometry
surface(x) = domain_z - ((x + 2000.0) / 1500.0)^2.0

# Build function for surface slope

dsdx_sym = Symbolics.derivative(surface(x), x)
dsdx_fn = Symbolics.build_function(dsdx_sym, x)
dsdx = eval(dsdx_fn)

fig = plot_surface_function(surface, dsdx)

#  ================================================
## Generate 2D (x, z) velocity field from SIA model
#  ================================================

u, w, dudx = sia_model((x, z), surface, dsdx)

to_plot = OrderedDict(
        ("u (Horizontal Velocity)", "u [m/a]") => (@. u(xs, zs')) * seconds_per_year,
        ("w (Vertical Velocity)", "w [m/a]") => (@. w(xs, zs')) * seconds_per_year,
        ("du/dx", "du/dx [a^-1]") => (@. dudx(xs, zs')) * seconds_per_year,
    )
;
fig = plot_fields(xs, zs, to_plot)

# w needs to be registered to be used in a PDESystem equation
# But it can't be registered after it's returned by a function, apparently
# 
# Approach 1: (kind of silly but simple workaround)
w_reg(x, z) = w(x, z)
@register w_reg(x, z)

# Approach 2: Create interpolation of w (probably saves compute time)
w_scipy = scipy_interpolate.RectBivariateSpline(xs, zs, (@. w(xs, zs')), bbox=(0, domain_x, 0, domain_z))
w_scipy_fn(x, z) = w_scipy(x, z)[1]
@register w_scipy_fn(x, z)

#  ============================================
## Run age-depth model and generate layer lines
#  ============================================

# Old version that applies zero age at fixed z
#age_xs, age_zs, age = age_depth((x, z), u, w_scipy_fn, domain_x, domain_z)

# Curvilinear grid solution applying zero age at surface(x)
age_xs, age_zs, age = age_depth_curvilinear((x, z), u, w_reg, domain_x, surface, dsdx,
                        fd_dq = 0.1, fd_dp = 500.0, output_dx = 100.0, output_dz = 40.0)

fig = plot_age(age_xs, age_zs, age, contour=false, colorrange=(0, 1000))
fig = plot_age(age_xs, age_zs, age)

layer_ages = 100:500:10000
layers, test = layers_from_age_depth(age_xs, age_zs, age, layer_ages)

begin
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1])
    for l in layers
        lines!(ax, xs, l(xs))
    end
    fig
end

#  ===
## Alternative layers based on particle flow
#  ===

layer_ages_tmp = 100:100:500
layers_t0 = repeat([surface], length(layer_ages_tmp))
u_meters_per_year(x, z) = u(x, z) * seconds_per_year
w_meters_per_year(x, z) = w_scipy_fn(x, z) * seconds_per_year


# TEST
xs = ages_xs
zs = @. surface(xs)
u0 = vcat(xs, zs)

function layer_velocity!(dxz, xz, p, t)
    # xy[1] is x, xy[2] is y
    dxz[1,:] = u_meters_per_year(xz[1,:], xz[2,:])
    dxz[2,:] = w_meters_per_year(xz[1,:], xz[2,:])
end

prob = ODEProblem(layer_velocity!, u0, (0.0, 10.0))

sol = solve(prob)

# TEST


layers = advect_layers(u_meters_per_year, w_meters_per_year, age_xs, layers_t0, layer_ages)

begin
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1])
    for l in layers
        lines!(ax, xs, l(xs))
    end
    fig
end

# TODO

#  ==========================
## Estimate layer deformation
#  ==========================

# TODO

#  =============================
## Solve for horizontal velocity
#  =============================

#sol = horizontal_velocity((x, z), d2l_dtdz, d2l_dxdz, dl_dx)
#fig = plot_horizontal_velocity_result(x, z, sol, layers, u)

# TODO


# @register_symbolic d2l_dtdz(x, z)
#     @register_symbolic d2l_dxdz(x, z)
#     @register dl_dx_scipy_fn(x, z)