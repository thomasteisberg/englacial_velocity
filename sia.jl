using CairoMakie
using ModelingToolkit
using Symbolics
using StatsFuns
using DifferentialEquations
using OrdinaryDiffEq
using Statistics
using OrderedCollections
using Integrals

using PyCall
scipy_interpolate = pyimport_conda("scipy.interpolate", "scipy")

## Setup domain, grid, and surface

# Domain size
domain_x = 10000 # meters
domain_z = 1500 # meters

# Discretization step size
dx = 100.0
dz = 50.0

xs = 0:dx:domain_x
zs = 0:dz:domain_z

# x, z are our spatial coordinates
# These will be used in various symbolic math throughout
@parameters x z

surface(x) = domain_z - ((x + 2000.0) / 1500.0)^2
#(1000 / 0.25) * logistic((x+200) / 5000) * (1 - logistic((x+200) / 5000))

dsdx_sym = Symbolics.derivative(surface(x), x)
dsdx_fn = Symbolics.build_function(dsdx_sym, x)
dsdx = eval(dsdx_fn)

begin
    dx_plot = dx
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

## Constants and temperature-dependent flow prefactor

ρ = 918 # kg/m^3
g = 9.8 # m/s^2
A0 = 3.985e-13 # s^-1 Pa^-3
Q = 60e3 # J mol^-1
R = 8.314 # J mol^-1 K^-1
seconds_per_year = 60 * 60 * 24 * 365

T_rel_p = 273.15 - 20

A = A0 * exp(-Q/(R * T_rel_p))

## Solve for u (horizontal velocity)

u(x, z) = -2 * A * dsdx(x)^3 * 0.25 * ρ^3 * g^3 * (surface(x)^4 - (z - surface(x))^4)

## Recover w (vertical velocity) through incompressibility

dudx_sym = Symbolics.derivative(u(x, z), x)
dudx_fn = Symbolics.build_function(dudx_sym, x, z)
dudx = eval(dudx_fn)

dwdz(x, z) = -1 * dudx(x, z)

# Numerical integration approach to find w(x, z) from incompressibility and a boundary condition
dwdz_inverted_parameters(z, x) = dwdz(x, z)
function w(x, z)
    prob = IntegralProblem(dwdz_inverted_parameters, 0.0, z, x)
    sol = solve(prob, QuadGKJL())
    return sol.u# + w0(x) # assume w(x, z=0) = 0 (no basal melt)
end

## Plot solutions

begin

    to_plot = OrderedDict(
        ("u (Horizontal Velocity)", "u [m/a]") => (@. u(xs, zs')) * seconds_per_year,
        ("w (Vertical Velocity)", "w [m/a]") => (@. w(xs, zs')) * seconds_per_year,
        ("du/dx", "du/dx [a^-1]") => (@. dudx(xs, zs')) * seconds_per_year,
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
    h = heatmap!(ax, xs, zs, (@. u(xs, zs')) * seconds_per_year)
    Colorbar(fig[1, 2], h, label="u [m/year]")

    fig
end