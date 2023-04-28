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
#  This doesn't work very well. If the grid it too coarse, the interpolation
#  to layers doesn't work very well. If the grid is too fine, the age-depth
#  model takes forever to run (and sometimes fails).

# # Old version that applies zero age at fixed z
# #age_xs, age_zs, age = age_depth((x, z), u, w_scipy_fn, domain_x, domain_z)

# # Curvilinear grid solution applying zero age at surface(x)
# fd_dq, fd_dp = 0.05, 500.0
# age_xs, age_zs, age = age_depth_curvilinear((x, z), u, w_reg, domain_x, surface, dsdx,
#                         fd_dq = fd_dq, fd_dp = fd_dp, output_dx = 100.0, output_dz = 10.0)

# #fig = plot_age(age_xs, age_zs, age, contour=false, colorrange=(0, 1000))
# fig = plot_age(age_xs, age_zs, age, title=title="Age-depth model\nfd_dq=$fd_dq, fd_dp=$fd_dp")

# layer_ages = 0:1000:10000
# layers, test = layers_from_age_depth(age_xs, age_zs, age, layer_ages)

# begin
#     fig = Figure(resolution=(1000, 300))
#     ax = Axis(fig[1, 1], title="Layers from age-depth model\nfd_dq=$fd_dq, fd_dp=$fd_dp")
#     for l in layers
#         lines!(ax, xs, l(xs))
#     end
#     ylims!(0,domain_z)
#     fig
# end

#  ===
## Alternative layers based on particle flow
#  ===

layer_ages = 0:250:20000
layers_t0 = advect_layer(u, w, xs, surface, layer_ages*seconds_per_year)

begin
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1], title="Layers from particle flow\n(advect_layers)")
    for l in layers_t0
        lines!(ax, xs, l(xs))
    end
    ylims!(0,domain_z)
    fig
end

#  ==========================
## Estimate layer deformation
#  ==========================

layers_t1 = Vector{Union{Function, PyObject}}(undef, length(layer_ages))

for i in 1:length(layer_ages)
    layers_t1[i] = advect_layer(u, w, xs, layers_t0[i], 1.0*seconds_per_year)[1]
end

begin
    deformation_xs = collect(Iterators.flatten(
        [collect(xs * 1.0) for i in eachindex(layers_t0)]))
    deformation_zs = collect(Iterators.flatten(
        [(layers_t1[i](xs) + layers_t0[i](xs)) / 2 for i in eachindex(layers_t0)]))
    deformation_delta_l = collect(Iterators.flatten(
        [layers_t1[i](xs) - layers_t0[i](xs) for i in eachindex(layers_t0)]))

    deformation_layer_slope = collect(Iterators.flatten(
        [(layers_t1[i](xs .+ 1.0) - layers_t1[i](xs) + layers_t0[i](xs .+ 1.0) - layers_t0[i](xs)) / 2 for i in eachindex(layers_t0)]))
end

# Interpolate scatter layer deformation and slope datapoints
# Choice of interpolation algorithm is hugely important here
# Eventually need to think carefully about the measurement error model and use
# that to select an apprpriate interpolation approach.
#
# SmoothBivariateSpline is nice in that it produces something tunably smooth
# and gives easy access to derivatives of the interpolator
dl_dt_scipy = scipy_interpolate.SmoothBivariateSpline(deformation_xs, deformation_zs, deformation_delta_l,
                                bbox=(0, domain_x, 0, domain_z))
dl_dt_scipy_fn(x, z) = dl_dt_scipy(x, z)[1] # Julia function wrapper
d2l_dtdz(x, z) = dl_dt_scipy.partial_derivative(0,1)(x, z)[1]

dl_dx_scipy = scipy_interpolate.SmoothBivariateSpline(deformation_xs, deformation_zs, deformation_layer_slope,
                                bbox=(0, domain_x, 0, domain_z))
dl_dx_scipy_fn(x, z) = dl_dx_scipy(x, z)[1] # Julia function wrapper
d2l_dxdz(x, z) = dl_dx_scipy.partial_derivative(0,1)(x, z)[1]


begin # Visualize layer vertical motion and verify interpolation
    clims = (-4, 4)
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1], title="Sampled points and interpolated dl_dt")
    h = heatmap!(ax, xs, zs, (@. dl_dt_scipy_fn(xs, zs')), colorrange=clims, colormap=:RdBu_5)
    Colorbar(fig[1, 2], h, label="Vertical Layer Deformation [m/yr]")
    scatter!(deformation_xs, deformation_zs, color=deformation_delta_l, colorrange=clims, colormap=:RdBu_5)
    fig
end

# Zoomed in visualization of just the top part. Useful to check boundary of interpolation
begin
    fig = Figure(resolution=(1000, 1000))

    clims=(-1.8, 1.8)

    zs_fine = 0:50:domain_z

    ax = Axis(fig[1, 1], title="dl / dt")
    h = contour!(ax, xs, zs_fine, (@. dl_dt_scipy_fn(xs, zs_fine')), colorrange=clims, levels=clims[1]:0.1:clims[2])
    scatter!(ax, deformation_xs, deformation_zs, color=deformation_delta_l, colorrange=clims)
    Colorbar(fig[1, 2], h, label="dl_dt [m/yr]")
    ylims!(1000,1500)
    fig
end


## >>> Visualize PDE input fields <<

# Input fields to our PDE:
# d2l_dtdz ✓
# d2l_dxdz ✓
# dl_dx ✓
# Not directly an input, but useful to see anyway:
# dl_dt ✓

begin
    fig = Figure(resolution=(1000, 1000))

    ax = Axis(fig[1, 1], title="d^2l / dtdz")
    h = heatmap!(ax, xs, zs, @. d2l_dtdz(xs, zs'))
    Colorbar(fig[1, 2], h, label="d2l_dtdz [m/(yr⋅m)]")

    ax = Axis(fig[2, 1], title="d^2l / dxdz")
    h = heatmap!(ax, xs, zs, @. d2l_dxdz(xs, zs'))
    Colorbar(fig[2, 2], h, label="d2l_dxdz [m/(m^2)]")

    ax = Axis(fig[3, 1], title="dl / dx")
    h = heatmap!(ax, xs, zs, (@. dl_dx_scipy_fn(xs, zs')), colorrange=(-0.3, 0.3), colormap=:RdBu_5)
    Colorbar(fig[3, 2], h, label="dl_dx [m/m]")

    ax = Axis(fig[4, 1], title="dl / dt")
    h = heatmap!(ax, xs, zs, (@. dl_dt_scipy_fn(xs, zs')), colorrange=(-4, 4), colormap=:RdBu_5)
    Colorbar(fig[4, 2], h, label="dl_dt [m/yr]")

    fig
end

to_plot = OrderedDict(
        ("d^2l / dtdz", "d2l_dtdz [m/(yr⋅m)]") => (@. d2l_dtdz(xs, zs')),
        ("d^2l / dxdz", "d2l_dxdz [m/(m^2)]") => (@. d2l_dxdz(xs, zs')),
        ("dl / dx", "dl_dx [m/m]") => (@. dl_dx_scipy_fn(xs, zs')),
        ("dl / dt", "dl_dt [m/yr]") => (@. dl_dt_scipy_fn(xs, zs'))
)

fig = plot_fields(xs, zs, to_plot)


#  =============================
## Solve for horizontal velocity
#  =============================

## >>> Setup and solve PDE <<<

# PDE we want to solve:
# d2l_dtdz + (u * d2l_dxdz) + (du_dz * dl_dx) + du_dx = 0
#
# Because u(x, z) is an already defined expression (representing "ground truth"),
# we'll call the thing we're estimating u_est(x, z)
# Rewritten:
# d2l_dtdz + (u_est * d2l_dxdz) + (Dz(u_est) * dl_dx) + Dx(u_est) ~ 0

@variables u_est(..)

# Spatial first derivative operators
Dx = Differential(x)
Dz = Differential(z)

@register_symbolic d2l_dtdz(x, z)
@register_symbolic d2l_dxdz(x, z)
@register dl_dx_scipy_fn(x, z)

# Our PDE
eq = [d2l_dtdz(x, z) + (u_est(x, z) * d2l_dxdz(x, z)) + (Dz(u_est(x, z)) * dl_dx_scipy_fn(x, z)) + Dx(u_est(x, z)) ~ 0]

# Boundary conditions
bcs = [u_est(0, z) ~ u(0, z), # Horizontal velocity at x=0 -- pretty much need this
       u_est(x, domain_z) ~ u(x, domain_z)] # Horizontal velocity along the surface -- less necessary -- inteesting to play with this

# Domain must be rectangular. Defined based on prior parameters
domains = [x ∈ Interval(0.0, domain_x),
           z ∈ Interval(0.0, domain_z)]

# x, z are independent variables. Solving for u_est(x, z)
@named pdesys = PDESystem(eq, bcs, domains, [x, z], [u_est(x, z)])

# Discretization step size
# Note: These MUST be floats. Easiest thing is just to add a ".0" :)
dx = 750.0
dz = 50.0

discretization = MOLFiniteDifference([x => dx, z => dz], nothing, approx_order=2)

prob = discretize(pdesys, discretization, progress=true)
sol = solve(prob, NewtonRaphson())

u_sol = sol[u_est(x, z)] # solver result on discretized grid

# Visualize result and compare with ground truth
begin
    u_true = (@. u(sol[x], sol[z]')) * seconds_per_year # Ground truth on PDE solution grid
    clims = (0, max(maximum(u_sol), maximum(u_true)))
    #clims = (0, maximum(u_true))

    fig = Figure(resolution=(1000, 1000))
    ax = Axis(fig[1, 1], title="PDE solution")
    h = heatmap!(ax, sol[x], sol[z], u_sol, colorrange=clims)
    Colorbar(fig[1, 2], h, label="Horizontal Velocity [m/yr]")

    # Ground truth for comparison
    ax = Axis(fig[2, 1], title="True values")
    h = heatmap!(ax, sol[x], sol[z], u_true, colorrange=clims)
    Colorbar(fig[2, 2], h, label="Horizontal Velocity [m/yr]")

    # Comparison between the two
    ax = Axis(fig[3, 1], title="solution - true values\n(layers shown as gray lines for reference)")
    h = heatmap!(ax, sol[x], sol[z], u_sol - u_true, colorrange=(-2, 2), colormap=:RdBu_5)
    Colorbar(fig[3, 2], h, label="Horizontal Velocity Difference [m/yr]")
    ylims!(ax, minimum(sol[z]), maximum(sol[z]))

    # Show layers for reference
    for (l_t0, l_t1) in zip(layers_t0, layers_t1)
        lines!(ax, xs, l_t0(xs), color=:gray, linestyle=:dash)
    end

    fig
end

#sol = horizontal_velocity((x, z), d2l_dtdz, d2l_dxdz, dl_dx)
#fig = plot_horizontal_velocity_result(x, z, sol, layers, u)

#
# TEST
#
#

surf_z = @. surface(age_xs)
u0 = vcat(age_xs', surf_z')

layers = Vector{Function}(undef, length(layer_ages_tmp))

function layer_velocity!(dxz, xz, p, t)
    # xz[1,:] is x, xz[2,:] is z
    dxz[1,:] = @. u_meters_per_year(xz[1,:], xz[2,:])
    dxz[2,:] = @. w_meters_per_year(xz[1,:], xz[2,:])
end

function simple_integration(u0, du_fn!::Function, t_end, dt)
    u = copy(u0)
    du = copy(u0)
    for t in 0:dt:t_end
        du_fn!(du, u, nothing, nothing)
        if sum(abs.(du) .> 10) > 0
            println(du)
            return nothing
        end
        u = u + du * dt
    end
    return u
end 

prob = ODEProblem(layer_velocity!, u0, (0.0, 1.0))
t0 = 0.0

for (layer_idx, layer_t) in enumerate(layer_ages_tmp)
    Δt = layer_t - t0
    
    prob = remake(prob, u0=u0, tspan=(0.0, Δt))
    sol = solve(prob)
    interp = scipy_interpolate.interp1d(sol.u[end][1,:], sol.u[end][2,:], kind="linear", fill_value="extrapolate")

    # println((layer_idx, layer_t))
    # u1 = simple_integration(u0, layer_velocity!, layer_t, 1)
    # interp = scipy_interpolate.interp1d(u1[1,:], u1[2,:], kind="linear", fill_value="extrapolate")

    layers[layer_idx] = interp

    t0 = layer_t

    layer_z = interp(age_xs)
    u0 = vcat(age_xs', layer_z')
end

begin
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1,1])
    lines!(ax, xs, (@. u_meters_per_year(xs, surface(xs))), label="u")
    lines!(ax, xs, (@. w_meters_per_year(xs, surface(xs))), label="w")
    fig
end

# TEST


layers = advect_layers(u_meters_per_year, w_meters_per_year, age_xs, surface, layer_ages)

begin
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1])
    for l in layers
        lines!(ax, xs, l(xs))
    end
    fig
end


to_plot = OrderedDict(
        ("u (Horizontal Velocity)", "u [m/a]") => (@. u_meters_per_year(xs, zs')),
        ("w (Vertical Velocity)", "w [m/a]") => (@. w_meters_per_year(xs, zs'))
    )
;
test_plot_xs = -1000.0:100.0:11000.0
test_plot_zs = -100.0:50.0:1600.0
fig = plot_fields(test_plot_xs, test_plot_zs, to_plot)

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