using Revise

using CairoMakie
using Symbolics
using OrderedCollections

includet("sia.jl")
includet("age_depth.jl")
includet("plot_helpers.jl")

seconds_per_year = 60 * 60 * 24 * 365.0 # m/s to m/yr conversion

## Problem definition and setup

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

## Generate 2D (x, z) velocity field from SIA model

u, w, dudx = sia_model((x, z), surface, dsdx)

to_plot = OrderedDict(
        ("u (Horizontal Velocity)", "u [m/a]") => (@. u(xs, zs')) * seconds_per_year,
        ("w (Vertical Velocity)", "w [m/a]") => (@. w(xs, zs')) * seconds_per_year,
        ("du/dx", "du/dx [a^-1]") => (@. dudx(xs, zs')) * seconds_per_year,
    )

fig = plot_fields(to_plot)

## Run age-depth model and generate layer lines

# TODO: Curvilinear grid for age depth

# w needs to be registered to be used in a PDESystem equation
# But it can't be registered after it's returned by a function, apparently
# This is an ugly workaround for now
# TODO: Figure out the right way to do this
w_reg = w(x, z)
@register w_reg(x, z)

age_xs, age_zs, age = age_depth((x, z), u, w_reg, domain_x, domain_z)

fig = plot_age(age_xs, age_zs, age)