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
domain_z = 1200.0 # meters

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

# Old version that applies zero age at fixed z
#age_xs, age_zs, age = age_depth((x, z), u, w_scipy_fn, domain_x, domain_z)

# Curvilinear grid solution applying zero age at surface(x)
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

layer_ages = 0:500:20000
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

dl_dt, d2l_dtdz, dl_dx, d2l_dxdz = estimate_layer_deformation(u, w, xs, layers_t0)

# begin # Visualize layer vertical motion and verify interpolation
#     clims = (-4, 4)
#     fig = Figure(resolution=(1000, 300))
#     ax = Axis(fig[1, 1], title="Sampled points and interpolated dl_dt")
#     h = heatmap!(ax, xs, zs, (@. dl_dt(xs, zs')), colorrange=clims, colormap=:RdBu_5)
#     Colorbar(fig[1, 2], h, label="Vertical Layer Deformation [m/yr]")
#     scatter!(deformation_xs, deformation_zs, color=deformation_delta_l, colorrange=clims, colormap=:RdBu_5)
#     fig
# end

# # Zoomed in visualization of just the top part. Useful to check boundary of interpolation
# begin
#     fig = Figure(resolution=(1000, 1000))

#     clims=(-1.8, 1.8)

#     zs_fine = 0:50:domain_z

#     ax = Axis(fig[1, 1], title="dl / dt")
#     h = contour!(ax, xs, zs_fine, (@. dl_dt(xs, zs_fine')), colorrange=clims, levels=clims[1]:0.1:clims[2])
#     scatter!(ax, deformation_xs, deformation_zs, color=deformation_delta_l, colorrange=clims)
#     Colorbar(fig[1, 2], h, label="dl_dt [m/yr]")
#     ylims!(1000,1500)
#     fig
# end


#  ==========================
## Visualize PDE input fields
#  ==========================

# Input fields to our PDE:
# d2l_dtdz ✓
# d2l_dxdz ✓
# dl_dx ✓
# Not directly an input, but useful to see anyway:
# dl_dt ✓

to_plot = OrderedDict(
        ("d^2l / dtdz", "d2l_dtdz [m/(yr⋅m)]") => (@. d2l_dtdz(xs, zs')),
        ("d^2l / dxdz", "d2l_dxdz [m/(m^2)]") => (@. d2l_dxdz(xs, zs')),
        ("dl / dx", "dl_dx [m/m]") => ((@. dl_dx(xs, zs')), Dict(:colorrange => (-0.3, 0.3), :colormap => :RdBu_5)),
        ("dl / dt", "dl_dt [m/yr]") => ((@. dl_dt(xs, zs')), Dict(:colorrange => (-4, 4), :colormap => :RdBu_5))
)

fig = plot_fields(xs, zs, to_plot)

#  =============================
## Solve for horizontal velocity
#  =============================

d2l_dtdz_reg(x, z) = d2l_dtdz(x, z)
@register d2l_dtdz_reg(x, z)
d2l_dxdz_reg(x, z) = d2l_dxdz(x, z)
@register d2l_dxdz_reg(x, z)
dl_dx_reg(x, z) = dl_dx(x, z)
@register dl_dx_reg(x, z)

xs_u, zs_u, u_est = horizontal_velocity((x, z), u, d2l_dtdz_reg, d2l_dxdz_reg, dl_dx_reg);
fig = plot_horizontal_velocity_result(xs_u, zs_u, u_est, layers_t0, u)
#fig = plot_horizontal_velocity_result(x, z, u_est, sol, layers_t0, u)

surface_velocity(x) = u(x, surface(x))
@register surface_velocity(x)
inflow_velocity(z) = u(0, z)
@register inflow_velocity(z)

xs_u, zs_u, u_est = horizontal_velocity_curvilinear((x, z), surface_velocity, inflow_velocity, surface, dsdx, d2l_dtdz_reg, d2l_dxdz_reg, dl_dx_reg; u_true=u);
fig = plot_horizontal_velocity_result(xs_u, zs_u, u_est, layers_t0, u)

#sol = horizontal_velocity_curvilinear((x, z), surface_velocity, inflow_velocity, surface, dsdx, d2l_dtdz_reg, d2l_dxdz_reg, dl_dx_reg; interpolate_to_xz=false, u_true=u);
#fig = plot_horizontal_velocity_result(sol[sol.ivs[1]], sol[sol.ivs[2]], sol[sol.dvs[1]], layers_t0, u)


