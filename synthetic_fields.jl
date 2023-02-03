using CairoMakie
using Dierckx
using DifferentialEquations
using DomainSets
using Interpolations
using MethodOfLines
using ModelingToolkit
using NonlinearSolve
using OrdinaryDiffEq
using SymbolicNumericIntegration
using Symbolics
using Integrals

## >>> Define domain and flow field parameters <<<

# Domain size
# Supports only rectangular domains for now (limitation primarily of MethodOfLines.jl)
# Must be declared const to avoid breaking some of the symbolic math
const domain_x = 10000 # meters
const domain_z = 1500 # meters

# x, z are our spatial coordinates
# These will be used in various symbolic math throughout
@parameters x z

# Define your desired horizontal flow field -- this is also ultimatey what we'll
# try to solve for
# We will automatically solve for the vertical component of the flow field, assuming
# that there is no convergence or divergence along the flow (v=0, dv/dy=0)
# This should be a valid Julia expression without case statements (if/else)
# (not strictly required, but things get more complicated...)
# Various cases you can try below:

# Linearly increasing in X and Z
u(x, z) = x .* (15 / domain_x) .+ z .* (10 / domain_z)
#u(x, z) = x .* (15 / domain_x) .+ z .* (2 / domain_z) .+ 8
# Increased sliding at the bed in the second half of the domain
# u(x, z) = (x .* (10 / domain_x) .+ z .* (5 / domain_z)) .+
#             ((20 ./ (1 .+ exp.(-0.001 .* (x .- 5000)))) .* (1 .- (z ./ domain_z)))
# Sinusoidal variations in bed speed, linearly damped to zero at the surface
# u(x, z) = (x .* (20 / domain_x) .+ z .* (5 / domain_z)) .+
#             (((3 .* sin.(2*π*x ./ 4000)) .+ 0) .* exp.(-z/50))

# A boundary condition for the vertical velocity is also needed. Zero velocity at the
# bed makes sense, but you could also make this negative to simulate basal melting.
# (This can also be a Julia expression.)
w0(x) = 0 # Boundary condition -- vertical velocity at bed

## >>> Calculate incompressibility-compatible flow field <<<
# Create functions for the vertical velocity field, using previously mentioned
# assumptions. Use Symbolics.jl math as far as they can take us, followed by
# plain old numerical integration.

dudx_sym = Symbolics.derivative(u(x, z), x)
dudx_fn = Symbolics.build_function(dudx_sym, x, z)
dudx = eval(dudx_fn)

dwdz(x, z) = -1 * dudx(x, z)

# SymbolicNumericIntegration approach -- sadly doesn't generalize well beacuse of
# https://github.com/SciML/SymbolicNumericIntegration.jl/issues/31
# w_sym, _, res = SymbolicNumericIntegration.integrate(dwdz(x, z), z)
# @assert res == 0 # res is the residual from the symbolic numeric interation. If non-zero, something went wrong
# w_indef = eval(build_function(w_sym, x, z))
# w_OLD(x, z) = w_indef(x, z) + w0(x) # Solved up to a constant. Add defined boundary condition

# Pure numerical integration approach
dwdz_inverted_parameters(z, x) = dwdz(x, z)
function w(x, z)
    prob = IntegralProblem(dwdz_inverted_parameters, 0, z, x)
    sol = solve(prob, QuadGKJL())
    return sol.u + w0(x)
end

# We specified u(x, z) (the horizontal velocity). We now have w(x, z) (the vertical velocity).
# We assume v(x, z) = 0

## >>> Create and simulate layers <<<

# Layers
# Layers are assumed to be straight lines at t=0 years
# Points are sampled from the initial layers, transported according to the
# specified flow field, and re-interpolated to generate the layers at t=1 years.
layer_slope_t0 = -0.01 # rise over run (i.e. meters/meter)
layer_spacing_t0 = 50 # meters -- vertical spacing between layers at x=0
# Number of layers to create. Layers are created starting layer_spacing_t0 meters from the surface
n_layers = 30

# Horizontal sampling spacing for when we perturb layers by transporting sampled particles
# from them according to the defined flow field.
layer_dx = 100
layer_interp_xs = 0:layer_dx:domain_x

# Define each layer at t=0 years as a linear interpolation
layers_t0 = [linear_interpolation(0:1, [0, layer_slope_t0] .+ (domain_z - (n * layer_spacing_t0)), extrapolation_bc=Line())
             for n = 1:n_layers]

# This is probably totally overkill, but...
# We discretize each layer to a series of points (spaced in x by layer_dx)
# and then, for each point, we numerically integrate its path from t=0 to t=1
# Finally, layers at t=1 year are defined by a linear interpolation of those points.

function layer_velocity!(dxy, xy, p, t)
    # xy[1] is x, xy[2] is y
    dxy[1] = u(xy[1], xy[2])
    dxy[2] = w(xy[1], xy[2])
end

# Dummy ODE problem defining the problem structure. We'll remake this problem
# with new initial conditions for each layer particle.
prob = ODEProblem(layer_velocity!, [0; 0], (0.0, 1.0))
begin
    layers_t1 = Array{Any}(undef, length(layers_t0))
    for layer_idx = eachindex(layers_t0)
        function prob_func(prob, i, repeat)
            x = (i - 1) * layer_dx
            remake(prob, u0=[x; layers_t0[layer_idx](x)])
        end

        layer_prob = EnsembleProblem(prob, prob_func=prob_func)
        sol = solve(layer_prob, trajectories=length(layer_interp_xs))

        transformed_layer_xs = [sol[i].u[end][1] for i = 1:length(layer_interp_xs)]
        transformed_layer_zs = [sol[i].u[end][2] for i = 1:length(layer_interp_xs)]
        layers_t1[layer_idx] = linear_interpolation(transformed_layer_xs, transformed_layer_zs, extrapolation_bc=Line())
    end
end

## >>> Visualize the layers and flow field <<<

# Sampling for visualization only
xs = (0:100:domain_x)
zs = 0:50:domain_z

begin
    us = u(xs, zs')
    ws = @. w(xs, zs')
    v_mag_max = maximum(sqrt.(us .^ 2 + ws .^ 2))
    v_scale = 500 / v_mag_max

    arrow_subsample = 5

    fig = Figure(resolution=(1000, 1000))
    ax = Axis(fig[1, 1])#, aspect=DataAspect())
    h = heatmap!(ax, xs, zs, sqrt.(us .^ 2 + ws .^ 2))
    Colorbar(fig[1, 2], h, label="Velocity Magnitude [m/yr]")
    arrows!(ax, xs[1:arrow_subsample:end], zs[1:arrow_subsample:end],
            v_scale * us[1:arrow_subsample:end, 1:arrow_subsample:end], v_scale * ws[1:arrow_subsample:end, 1:arrow_subsample:end])
    for (l_t0, l_t1) in zip(layers_t0, layers_t1)
        lines!(xs, l_t0(xs), color=:blue)
        lines!(xs, l_t1(xs), color=:red, linestyle=:dash)
    end
    xlims!(0, domain_x)
    ylims!(0, domain_z)

    ax = Axis(fig[2, 1])
    h = heatmap!(ax, xs, zs, us, colormap=:RdBu_5, colorrange=(-v_mag_max, v_mag_max))
    Colorbar(fig[2, 2], h, label="Horizontal Velocity [m/yr]")
    xlims!(ax, 0, domain_x)
    ylims!(ax, 0, domain_z)

    ax = Axis(fig[3, 1])
    ws_abs_max = max(abs(maximum(ws)), abs(minimum(ws)))
    h = heatmap!(ax, xs, zs, ws, colormap=:RdBu_5, colorrange=(-ws_abs_max, ws_abs_max))
    Colorbar(fig[3, 2], h, label="Vertical Velocity [m/yr]")
    xlims!(ax, 0, domain_x)
    ylims!(ax, 0, domain_z)

    fig
end

## >>> Estimate layer deformation dl(x,z)/dt <<<

begin
    deformation_xs = collect(Iterators.flatten(
        [collect(layer_interp_xs * 1.0) for i in eachindex(layers_t0)]))
    deformation_zs = collect(Iterators.flatten(
        [(layers_t1[i](layer_interp_xs) + layers_t0[i](layer_interp_xs)) / 2 for i in eachindex(layers_t0)]))
    deformation_delta_l = collect(Iterators.flatten(
        [layers_t1[i](layer_interp_xs) - layers_t0[i](layer_interp_xs) for i in eachindex(layers_t0)]))

    deformation_layer_slope = collect(Iterators.flatten(
        [[(Interpolations.gradient(layers_t0[i], x)[1] + Interpolations.gradient(layers_t1[i], x)[1])/2 for x = layer_interp_xs] for i in eachindex(layers_t0)]))

    # Some of the layers go above the ice surface or below the bed -- clip those out
    layer_in_ice_mask = (deformation_zs .< domain_z) .& (deformation_zs .> 0)

    deformation_xs = deformation_xs[layer_in_ice_mask]
    deformation_zs = deformation_zs[layer_in_ice_mask]
    deformation_delta_l = deformation_delta_l[layer_in_ice_mask]

    deformation_layer_slope = deformation_layer_slope[layer_in_ice_mask]
end

# Interpolations.jl doesn't support scattered interpolation, so use Dierckx for this
dl_dt = Dierckx.Spline2D(deformation_xs, deformation_zs, deformation_delta_l, s=1e-0)
dl_dx = Dierckx.Spline2D(deformation_xs, deformation_zs, deformation_layer_slope, s=1e-0)
finite_diff_half_length = 1.0
# Have dl/dt and dl/dx. Just need to calculate the second derivatives.
# Currently a manual finite difference. Should probably do something better.
d2l_dxdz(x, z) = (dl_dx(x, z .+ finite_diff_half_length) - dl_dx(x, z .- finite_diff_half_length)) / (2 * finite_diff_half_length)
d2l_dtdz(x, z) = (dl_dt(x, z .+ finite_diff_half_length) - dl_dt(x, z .- finite_diff_half_length)) / (2 * finite_diff_half_length)

begin # Visualize layer vertical motion and verify interpolation
    clims = (-4, 4)
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1], title="Sampled points and interpolated dl_dt")
    h = heatmap!(ax, xs, zs, evalgrid(dl_dt, xs, zs), colorrange=clims, colormap=:RdBu_5)
    Colorbar(fig[1, 2], h, label="Vertical Layer Deformation [m/yr]")
    scatter!(deformation_xs, deformation_zs, color=deformation_delta_l, colorrange=clims, colormap=:RdBu_5)
    fig
end

## >>> Show single layer deformation <<<

begin
    layer_idx = round(Int, length(layers_t0)/2)

    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1])

    plot!(ax, xs, layers_t0[layer_idx](xs))
    plot!(ax, xs, layers_t1[layer_idx](xs))

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
    h = heatmap!(ax, xs, zs, @. dl_dx(xs, zs'))
    Colorbar(fig[3, 2], h, label="dl_dx [m/m]")

    ax = Axis(fig[4, 1], title="dl / dt")
    h = heatmap!(ax, xs, zs, @. dl_dt(xs, zs'))
    Colorbar(fig[4, 2], h, label="dl_dt [m/yr]")

    fig
end

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

dl_dx_fn(x, z) = dl_dx(x, z)
@register dl_dx_fn(x, z) # An ugly hack -- dl_dx is a Dierckx.Spline2D, which can't be registered directly

# Our PDE
eq = [d2l_dtdz(x, z) + (u_est(x, z) * d2l_dxdz(x, z)) + (Dz(u_est(x, z)) * dl_dx_fn(x, z)) + Dx(u_est(x, z)) ~ 0]

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
sol = solve(prob) # TODO: Think about appropriate solver algorithm. Auto-selection picks different options.

u_sol = sol[u_est(x, z)] # solver result on discretized grid

# Visualize result and compare with ground truth
begin
    u_true = u(sol[x], sol[z]') # Ground truth on PDE solution grid
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
    h = heatmap!(ax, sol[x], sol[z], u_sol - u_true, colorrange=(-5, 5), colormap=:RdBu_5)
    Colorbar(fig[3, 2], h, label="Horizontal Velocity Difference [m/yr]")
    ylims!(ax, minimum(sol[z]), maximum(sol[z]))

    # Show layers for reference
    for (l_t0, l_t1) in zip(layers_t0, layers_t1)
        lines!(ax, xs, l_t0(xs), color=:gray, linestyle=:dash)
    end

    fig
end