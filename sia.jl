using CairoMakie
using ModelingToolkit
using Symbolics
using StatsFuns
using DifferentialEquations
using OrdinaryDiffEq
using Statistics
using OrderedCollections
using Integrals

# TODO: Remaining issue: Consider top surface geometry

# ## Setup domain, grid, and surface

# # Domain size
# domain_x = 10000.0 # meters
# domain_z = 1500.0 # meters

# # Discretization step size
# dx = 100.0
# dz = 50.0

# xs = 0.0:dx:domain_x
# zs = 0.0:dz:domain_z

# # x, z are our spatial coordinates
# # These will be used in various symbolic math throughout
# @parameters x z

# surface(x) = domain_z - ((x + 2000.0) / 1500.0)^2.0

# dsdx_sym = Symbolics.derivative(surface(x), x)
# dsdx_fn = Symbolics.build_function(dsdx_sym, x)
# dsdx = eval(dsdx_fn)

function plot_surface_function(surface::Function, dsdx::Function, dx_plot = 100.0)
    xs_plot = 0:dx_plot:domain_x

    fig = Figure(resolution=(1000, 600))
    ax = Axis(fig[1, 1])
    lines!(ax, xs_plot, (@. surface(xs_plot)), label="Surface elevation")
    axislegend()

    ax = Axis(fig[2,1])
    lines!(ax, xs_plot, (@. dsdx(xs_plot)), label="ds/dx")
    axislegend()

    return fig
end

# ## Constants and temperature-dependent flow prefactor

function sia_model(spatial_parameters::Tuple{Num, Num}, surface::Function, dsdx::Function;
    ρ::Float64 = 918.0, g::Float64 = 9.8, A0::Float64 = 3.985e-13, n_A0::Float64 = 3.0,
    Q::Float64 = 60.0e3, R::Float64 = 8.314, T_rel_p::Float64 = (273.15-20), n::Float64 = 3.0,
    basal_velocity::Function = (x) -> 0.0)

    # Implementation is based on Section 3.4 of Ralf Greve's course notes:
    # https://ocw.hokudai.ac.jp/wp-content/uploads/2016/02/DynamicsOfIce-2005-Note-all.pdf

    # Constant defaults and units:
    # ρ = 918.0 # kg/m^3
    # g = 9.8 # m/s^2
    # A0 = 3.985e-13 # s^-1 Pa^-3
    # Q = 60.0e3 # J mol^-1
    # R = 8.314 # J mol^-1 K^-1
    # T_rel_p = 273.15 - 20

    x, z = spatial_parameters

    # Baseline case A value (i.e. the exponent for which )
    A = A0 * exp(-Q/(R * T_rel_p))

    if n != n_A0
        println("Re-calculating A for requested exponent n = $n")
        log_ref_stress = 6.0
        log_ref_strain_rate = log10.(2A) .+ n_A0 * log_ref_stress
        log_2A = log_ref_strain_rate - n * log_ref_stress # log10(2A)
        A = 10^(log_2A) / 2
    end

    println("n = $n, \tA = $A")

    # Solve for u (horizontal velocity)

    u(x, z) = (-2.0 * A * abs(dsdx(x))^(n-1.0) * dsdx(x) * ρ^n * g^n * (surface(x)^(n+1.0) - max(surface(x) - z, 0)^(n+1.0)) / (n + 1.0)) + basal_velocity(x)

    # Recover w (vertical velocity) through incompressibility

    dudx_sym = Symbolics.derivative(u(x, z), x)
    dudx_fn = Symbolics.build_function(dudx_sym, x, z)
    dudx = eval(dudx_fn)

    dwdz(x, z) = -1 * dudx(x, z)

    # Numerical integration approach to find w(x, z) from incompressibility and a boundary condition
    dwdz_inverted_parameters(z, x) = dwdz(x, z)
    function w(x, z)
        prob = IntegralProblem(dwdz_inverted_parameters, 0.0, z, x)
        sol = solve(prob, QuadGKJL())
        return sol.u # + w0(x) # assume w(x, z=0) = 0 (no basal melt)
    end

    return u, w, dudx
end

# ## Plot solutions

# begin

#seconds_per_year = 60 * 60 * 24 * 365.0
#     to_plot = OrderedDict(
#         ("u (Horizontal Velocity)", "u [m/a]") => (@. u(xs, zs')) * seconds_per_year,
#         ("w (Vertical Velocity)", "w [m/a]") => (@. w(xs, zs')) * seconds_per_year,
#         ("du/dx", "du/dx [a^-1]") => (@. dudx(xs, zs')) * seconds_per_year,
#     )

#     fig = Figure(resolution=(1000, 300*length(to_plot)))

#     for (idx, titles) in enumerate(keys(to_plot))
#         ax = Axis(fig[idx, 1], title=titles[1])
#         h = heatmap!(ax, xs, zs, to_plot[titles])#, colorrange=(-1,1))
#         cb = Colorbar(fig[idx, 2], h, label=titles[2])
#     end

#     fig
# end

# ## Age depth model

# using MethodOfLines
# using DomainSets
# using NonlinearSolve

# # ENV["PYTHON"] = ""
# # using PyCall
# # scipy_interpolate = pyimport_conda("scipy.interpolate", "scipy")

# # u_scipy = scipy_interpolate.RectBivariateSpline(xs, zs, @. u(xs, zs'),
# #                                 bbox=(0, domain_x, 0, domain_z))

# @variables age(..)

# # Spatial first derivative operators
# Dx = Differential(x)
# Dz = Differential(z)

# #@register u(x, z)
# @register w(x, z)

# # Our PDE
# #eq = [u(x,z) * age(x, z) ~ 0.0]
# eq = [seconds_per_year * u(x,z) * Dx(age(x, z)) + seconds_per_year * w(x, z) * Dz(age(x, z)) ~ 1.0]
# #eq = [Dz(age(x, z)) ~ 1.0]

# # Boundary conditions
# bcs = [
#     age(x, domain_z) ~ 0,
#     #age(0, z) ~ 1000
#     ]

# # Domain must be rectangular. Defined based on prior parameters
# domains = [x ∈ Interval(0.0, domain_x),
#            z ∈ Interval(0.0, domain_z)]

# # x, z are independent variables. Solving for u_est(x, z)
# @named pdesys = PDESystem(eq, bcs, domains, [x, z], [age(x, z)])

# # Discretization step size
# # Note: These MUST be floats. Easiest thing is just to add a ".0" :)
# #fd_dx, fd_dz = 2000.0, 200.0
# fd_dx = 500.0
# fd_dz = 50.0

# discretization = MOLFiniteDifference([x => fd_dx, z => fd_dz], nothing)

# prob = discretize(pdesys, discretization, progress=true)
# sol = solve(prob, NewtonRaphson())

# age_sol = sol[age(x, z)] # solver result on discretized grid

# begin
#     fig = Figure(resolution=(1000, 300))

#     ax = Axis(fig[1, 1], title="age")
#     h = contour!(ax, sol[x], sol[z], age_sol,
#                     levels=100:200:10000,
#                     colorrange=(0,10000)
#                 )
#     cb = Colorbar(fig[1, 2], h, label="Years")
    
#     fig
# end