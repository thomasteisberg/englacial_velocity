using CairoMakie
using ModelingToolkit
using NonlinearSolve
using Symbolics
using StatsFuns
using DomainSets
using DifferentialEquations
using OrdinaryDiffEq
using Statistics
using OrderedCollections

using PyCall
scipy_interpolate = pyimport_conda("scipy.interpolate", "scipy")

## Setup domain, grid, and surface

# Domain size
# Must be declared const to avoid breaking some of the symbolic math
const domain_x = 1000 # meters
const domain_z = 100 # meters

# Discretization step size
dx = 20.0
dz = 5.0

xs = 0:dx:domain_x
zs = 0:dz:domain_z

# x, z are our spatial coordinates
# These will be used in various symbolic math throughout
@parameters x z

surface(x) = (1000 / 0.25) * logistic(x / 500) * (1 - logistic(x / 500))
surface_velocity(x) = (5 + 10*(x/domain_x)) / (60*60*24*365)

dsdx_sym = Symbolics.derivative(surface(x), x)
dsdx_fn = Symbolics.build_function(dsdx_sym, x)
dsdx = eval(dsdx_fn)

begin
    dx_plot = 50
    xs_plot = 0:dx_plot:domain_x

    fig = Figure(resolution=(1000, 600))
    ax = Axis(fig[1, 1])
    lines!(ax, xs_plot, (@. surface(xs_plot)), label="Surface elevation")
    axislegend()

    ax = Axis(fig[2,1])
    lines!(ax, xs_plot, (@. dsdx(xs_plot)), label="ds/dx")
    axislegend()

    fig
end

## Define initial grid of μ (effective ice viscosity)

B = 1.419e8 # s^(1/3) Pa -- from cuffey(263.15)
n = 3
ρ = 918 # kg/m^3
g = 9.8 # m/s^2

tmp_dudx = 0.000001
tmp_dudz = 0.000001
ϵe = sqrt(tmp_dudx^2 + (1/4)*tmp_dudz^2)
eff_visc = B / (2 * ϵe^((1-n)/n))

μ_grid = eff_visc * ones(Float32, (length(xs), length(zs)))

max_iters = 40
max_abs_change_μ = zeros(Float32, (max_iters,))
rms_change_μ = zeros(Float32, (max_iters,))

for iter_idx = 1:max_iters

    ## Repeat until convergence:
    ## |     Create interpolation of μ

    μ_interp = scipy_interpolate.RectBivariateSpline(xs, zs, μ_grid)
    μ_interp_fn(x, z) = μ_interp(x, z)[1] # Julia function wrapper
    dμ_dz(x, z) = μ_interp.partial_derivative(0,1)(x, z)[1]

    ## |     For each x, integrate to find horizontal velocities
    ## |      and sample horizontal velocities into grid

    function sia_known_μ_bvp!(du, u, x, z)
        vx = u[1]
        dvx = u[2]
        du[1] = dvx
        du[2] = (ρ * g * dsdx(x) - (dμ_dz(x, z) * dvx)) / μ_interp_fn(x, z)
    end
    function bc(residual, u, x, z)
        residual[1] = u[1][1]
        residual[2] = u[end][1] - surface_velocity(x)
    end
    function sia_known_μ_iv(du, u, x, z)
        (ρ * g * dsdx(x) - (dμ_dz(x, z) * du)) / μ_interp_fn(x, z)
    end

    u_grid = zeros(Float32, (length(xs), length(zs)))

    for (idx, x) in enumerate(xs)
        prob = SecondOrderODEProblem(sia_known_μ_iv, 0, 0, (0, domain_z), x)
        #prob = TwoPointBVProblem(sia_known_μ_bvp!, bc, [0, 0], (0, domain_z), x)
        sol = solve(prob)

        u = [sol.u[idx][1] for idx = 1:length(sol.u)]
        u_grid[idx, :] = scipy_interpolate.griddata(sol.t, u, zs)
    end

    ## |     Calculate vertical velocities from incompressibility

    u_interp = scipy_interpolate.RectBivariateSpline(xs, zs, u_grid)
    du_dx(x, z) = u_interp.partial_derivative(1,0)(x, z)
    du_dz(x, z) = u_interp.partial_derivative(0,1)(x, z)

    ## |     Calculate μ

    ϵe = 0.5 * (du_dx(xs, zs)).^2 + 0.25 * (du_dz(xs, zs)).^2
    μ_grid_new = B ./ (2 * ϵe.^((1-n)/n))

    ## |     Check for convergence of μ and quit if satisfied

    max_abs_change_μ[iter_idx] = maximum(abs.(μ_grid_new - μ_grid))
    rms_change_μ[iter_idx] = sqrt(mean((μ_grid_new - μ_grid).^2))

    μ_grid = 0.9 * μ_grid + 0.1 * μ_grid_new

end

begin
    fig = Figure(resolution=(1000, 600))
    ax = Axis(fig[1, 1], title="Max absolute change in μ")
    plot!(1:max_iters, max_abs_change_μ)
    ax = Axis(fig[2, 1], title="RMS change in μ")
    plot!(1:max_iters, rms_change_μ)

    fig
end

max_abs_change_μ, rms_change_μ


begin

    to_plot = OrderedDict(
        ("μ (Effective Viscosity)", "μ [Pa s]") => μ_grid,
        ("u (Horizontal Velocity)", "u [m/s]") => u_grid,
        ("du/dx", "du/dx [s^-1]") => du_dx(xs, zs),
        ("du/dz", "du/dz [s^-1]") => du_dz(xs, zs)
    )

    fig = Figure(resolution=(1000, 300*length(to_plot)))

    for (idx, titles) in enumerate(keys(to_plot))
        ax = Axis(fig[idx, 1], title=titles[1])
        h = heatmap!(ax, xs, zs, to_plot[titles])#, colorrange=(-1,1))
        Colorbar(fig[idx, 2], h, label=titles[2])
    end

    fig
end

begin # TODO move this

    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1], title="u")
    h = heatmap!(ax, xs, zs, u_grid)
    Colorbar(fig[1, 2], h, label="u [m/s]")

    fig
end