using Revise

using CairoMakie
using Colors
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
#surface(x) = domain_z - ((x + 2000.0) / 1500.0)^2.0
surface(x) = domain_z - (x / 1000.0)^2.0

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

# # Approach 2: Create interpolation of w (probably saves compute time)
# w_scipy = scipy_interpolate.RectBivariateSpline(xs, zs, (@. w(xs, zs')), bbox=(0, domain_x, 0, domain_z))
# w_scipy_fn(x, z) = w_scipy(x, z)[1]
# @register w_scipy_fn(x, z)

fig

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

layer_ages = 0:100:5000
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

dl_dt, d2l_dtdz, dl_dx, d2l_dxdz, deformation_debug = estimate_layer_deformation(u, w, xs, layers_t0)

dl_dt, d2l_dtdz, dl_dx, d2l_dxdz, deformation_debug = estimate_layer_deformation_monotone(u, w, 0:200:domain_x, 0:10:domain_z, layers_t0)

begin # Visualize layer vertical motion and verify interpolation
    clims = (-4, 4)
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1], title="Sampled points and interpolated dl_dt")
    h = heatmap!(ax, xs, zs, (@. dl_dt(xs, zs')), colorrange=clims, colormap=:RdBu_5)
    Colorbar(fig[1, 2], h, label="Vertical Layer Deformation [m/yr]")
    scatter!(deformation_debug["deformation_xs"], deformation_debug["deformation_zs"], color=deformation_debug["deformation_delta_l"], colorrange=clims, colormap=:RdBu_5)
    fig
end

begin # Visualize layer vertical motion and verify interpolation
    clims = (-0.5, 0.5)
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1], title="Sampled points and interpolated dl_dx")
    h = heatmap!(ax, xs, zs, (@. dl_dx(xs, zs')), colorrange=clims, colormap=:RdBu_5)
    Colorbar(fig[1, 2], h, label="Layer Slope [m/m]")
    scatter!(deformation_debug["deformation_xs"], deformation_debug["deformation_zs"],
             color=deformation_debug["deformation_layer_slope"], colorrange=clims, colormap=:RdBu_5)
    fig
end

begin # Visualize layer vertical motion and verify interpolation
    clims = (-0.008, 0.008)
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1], title="Interpolated d2l_dtdz")
    h = heatmap!(ax, xs, zs, (@. d2l_dtdz(xs, zs')), colorrange=clims, colormap=:RdBu_5)
    Colorbar(fig[1, 2], h, label="")
    #scatter!(deformation_debug["deformation_xs"], deformation_debug["deformation_zs"], color=deformation_debug["deformation_delta_l"], colorrange=clims, colormap=:RdBu_5)
    fig
end

begin # Visualize layer vertical motion and verify interpolation
    clims = (-0.002, 0.002)
    fig = Figure(resolution=(1000, 300))
    ax = Axis(fig[1, 1], title="Interpolated d2l_dxdz")
    h = heatmap!(ax, xs, zs, (@. d2l_dxdz(xs, zs')), colorrange=clims, colormap=:RdBu_5)
    Colorbar(fig[1, 2], h, label="")
    #scatter!(deformation_debug["deformation_xs"], deformation_debug["deformation_zs"], color=deformation_debug["deformation_delta_l"], colorrange=clims, colormap=:RdBu_5)
    fig
end

begin # Plot dl_dt and d2l_dtdz for a single x value
    fig = Figure(resolution=(1000, 300))
    x_pos = 5000
    ax = Axis(fig[1, 1], title="dl_dt and d2l_dtdz for x = $x m")
    ax2 = Axis(fig[1,1])
    lines!(ax, zs, (@. dl_dt(x, zs)), label="dl_dt")
    ax.ylabel = "dl_dt"
    ax.xlabel = "z [m]"

    #x_idx = argmin( abs.(deformation_debug["deformation_xs"] - x))
    mask = deformation_debug["deformation_xs"] .== x_pos

    plot!(ax, deformation_debug["deformation_zs"][mask], deformation_debug["deformation_delta_l"][mask],
        color=:blue)

    lines!(ax2, zs, (@. d2l_dtdz(x_pos, zs)), label="d2l_dtdz", color=:red)

    ax2.yaxisposition = :right
    ax2.yticklabelalign = (:left, :center)
    ax2.xticklabelsvisible = false
    ax2.xticklabelsvisible = false
    ax2.xlabelvisible = false
    ax2.ylabel = "d2l_dtdz"
    ax2.ytickcolor = ax2.yticklabelcolor = ax2.ylabelcolor = :red
    ax2.ygridcolor = RGBA(Colors.color_names["red"]..., 0.5)
    ax2.ygridstyle = :dash

    linkxaxes!(ax,ax2)

    fig
end

begin # Plot dl_dx and d2l_dxdz for a single x value
    fig = Figure(resolution=(1000, 300))
    x_pos = 4800
    ax = Axis(fig[1, 1], title="dl_dx and d2l_dxdz for x = $x m")
    ax2 = Axis(fig[1,1])
    lines!(ax, zs, (@. dl_dx(x_pos, zs)), label="dl_dx")
    ax.ylabel = "dl_dx"
    ax.xlabel = "z [m]"

    mask = deformation_debug["deformation_xs"] .== x_pos

    plot!(ax, deformation_debug["deformation_zs"][mask], deformation_debug["deformation_layer_slope"][mask],
        color=:blue)

    lines!(ax2, zs, (@. d2l_dxdz(x_pos, zs)), label="d2l_dxdz", color=:red)

    ax2.yaxisposition = :right
    ax2.yticklabelalign = (:left, :center)
    ax2.xticklabelsvisible = false
    ax2.xticklabelsvisible = false
    ax2.xlabelvisible = false
    ax2.ylabel = "d2l_dxdz"
    ax2.ytickcolor = ax2.yticklabelcolor = ax2.ylabelcolor = :red
    ax2.ygridcolor = RGBA(Colors.color_names["red"]..., 0.5)
    ax2.ygridstyle = :dash

    linkxaxes!(ax,ax2)

    fig
end

1+1

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

#xs_u, zs_u, u_est = horizontal_velocity_curvilinear((x, z), surface_velocity, inflow_velocity, surface, dsdx, d2l_dtdz_reg, d2l_dxdz_reg, dl_dx_reg; u_true=u);
#fig = plot_horizontal_velocity_result(xs_u, zs_u, u_est, layers_t0, u)

dp, dq = 500.0, 100.0/1200.0
sol = horizontal_velocity_curvilinear((x, z), surface_velocity, inflow_velocity, surface, dsdx, d2l_dtdz_reg, d2l_dxdz_reg, dl_dx_reg;
    interpolate_to_xz=false, u_true=u, dp = dp, dq = dq);
#fig = plot_horizontal_velocity_result(sol[sol.ivs[1]], sol[sol.ivs[2]], sol[sol.dvs[1]], layers_t0, u)
fig = plot_horizontal_velocity_result(sol[sol.ivs[1]], surface(0)*sol[sol.ivs[2]], sol[sol.dvs[1]], layers_t0, u)

begin
    p, q = sol.ivs
    X = ones(length(sol[q]))' .* sol[p]
    Q = sol[q]' .* ones(length(sol[p]))
    Z = Q .* (@. surface(X))

    output_dx, output_dz = 200.0, 50.0

    x_out = 0:output_dx:domain_x
    z_out = 0:output_dz:maximum(Z)
    X_out = ones(length(z_out))' .* x_out
    Z_out = z_out' .* ones(length(x_out))

    X_vec = vec(X)
    Z_vec = vec(Z)
    Q_vec = vec(Q)

    mask = .!( ((X_vec .== 0) .| (X_vec .== domain_x)) .& ((Q_vec .== 0) .| (Q_vec .== 1.0)) )

    u_est_grid = scipy_interpolate.griddata((X_vec[mask], Z_vec[mask]),
                    vec(sol[sol.dvs[1]])[mask],
                    (X_out, Z_out),
                    method="linear")

    xs_u, zs_u, u_est = x_out, z_out, u_est_grid
end

begin # Plot results from curvilinear grid along with top boundary condition
    X = ones(length(sol[sol.ivs[2]]))' .* sol[sol.ivs[1]]
    Q = sol[sol.ivs[2]]' .* ones(length(sol[sol.ivs[1]]))
    Z = Q .* (@. surface(X))
    clims = (0, 25)
    fig = Figure(resolution=(1000, 500))
    ax = Axis(fig[1, 1], title="Curvilinear Grid PDE Output (dp = $dp, dq = $dq)")
    #h = heatmap!(ax, xs, zs, (@. dl_dx(xs, zs')), colorrange=clims, colormap=:RdBu_5)
    #Colorbar(fig[1, 2], h, label="Layer Slope [m/m]")
    h = heatmap!(ax, xs_u, zs_u, u_est, colorrange=clims)
    sc = scatter!(ax, vec(X), vec(Z), color=vec(sol[sol.dvs[1]]), colorrange=clims)

    # Top boundary condition -- offset up by 100m for clarity
    scatter!(ax, sol[sol.ivs[1]], (@. surface(sol[sol.ivs[1]])) .+ 100, color=(@. surface_velocity(sol[sol.ivs[1]])) .* seconds_per_year,
            colorrange=clims, marker=:x)

    Colorbar(fig[1, 2], sc, label="Horizontal Velocity [m/yr]")

    Z_diff = (Z[:, 1:end-1] .+ Z[:, 2:end] ) ./ 2
    X_diff = X[:, 1:end-1]

    scatter!(ax, vec(X_diff), vec(Z_diff), color=vec(diff(sol[sol.dvs[1]], dims=2)), colorrange=(-0.5, 0.5), colormap=:RdBu_5, markersize=5, marker=:rect)

    fig
end

fig

#  =============================
## Estimate ice effective viscosity
#  =============================

# Find the finite difference derivatives of u_est with respect to z
u_true = @. u(xs_u, zs_u')# * seconds_per_year
dudz_fd_true = diff(u_true, dims=2) ./ diff(zs_u)'
dudz_fd = diff(u_est ./ seconds_per_year, dims=2) ./ diff(zs_u)'
ρ = 918 # kg/m^3
g = 9.81 # m/s^2

dist_to_surf = (@. surface(xs_u)) .- zs_u'
diff_tmp = diff(dist_to_surf, dims=2)
dist_to_surf_fd = dist_to_surf[:, 1:end-1] .+ (diff_tmp ./ 2)

#valid_regions_mask = ((xs_u .> 2000.0) .& (xs_u .< (domain_x - 3000))) .* ((zs_u[1:end-1] .> 400) .& (zs_u[1:end-1] .< (domain_z-400)))'
valid_regions_mask = ((xs_u .> 1000.0) .& (xs_u .< (domain_x - 1000))) .* ((zs_u[1:end-1] .> 200) .& (zs_u[1:end-1] .< (domain_z-200)))'

rheology_eff_stress = (ρ * g * dist_to_surf_fd .* -(@. dsdx(xs_u))) # (3.88)
rheology_log_eff_stress = log10.( rheology_eff_stress[valid_regions_mask] )
rheology_log_eff_strain = log10.( max.((dudz_fd[valid_regions_mask]), 1e-22) )
#rheology_log_eff_strain = log10.( (dudz_fd_true[valid_regions_mask]) )

# test -- stress
A = 1.658286764403978e-25
x_pos, z_pos = 5000, 900
eff_stress = -(dsdx(x_pos) * ρ * g * (surface(x_pos) - z_pos))
x_idx, z_idx = argmin(abs.(xs_u .- x_pos)), argmin(abs.(zs_u .- z_pos))
rheology_eff_stress[x_idx, z_idx]
(rheology_eff_stress[x_idx, z_idx]) / eff_stress

# test - strain rate
dudz_test = 2 * A * eff_stress^3
dudz_fd_true[x_idx, z_idx]
dudz_fd_true[x_idx, z_idx] / dudz_test

# test
err = ((2 * A * rheology_eff_stress .^ 3) ./ dudz_fd_true)

begin
    fig = Figure(resolution=(1000, 1000))
    ax = Axis(fig[1, 1], title="Effective stress vs strain rate")
    scatter!(ax, rheology_log_eff_stress, rheology_log_eff_strain)
    stress_xs = 0:0.1:5.5
    #lines!(ax, stress_xs, 3 .* stress_xs .- 39.1, linestyle=:dash, color=:black)
    lines!(ax, stress_xs, 3 .* stress_xs .+ log10(2*A), linestyle=:dash, color=:black)
    #xlims!(-15, -5)
    fig
end

masked_dudz_fd = copy(dudz_fd)
masked_dudz_fd[.! valid_regions_mask] .= 0.0
masked_dudz_fd_true = copy(dudz_fd_true)
masked_dudz_fd_true[.! valid_regions_mask] .= 0.0

to_plot = OrderedDict(
        ("dudz_fd (estimated)", "dudz") => (masked_dudz_fd, Dict(:colorrange => (-0.03, 0.03), :colormap => :RdBu_5)),
        ("dudz_fd (true)", "dudz") => (masked_dudz_fd_true, Dict(:colorrange => (-0.03, 0.03), :colormap => :RdBu_5)),
        ("difference", "") => (masked_dudz_fd - masked_dudz_fd_true, Dict(:colorrange => (-0.005, 0.005), :colormap => :RdBu_5))
)

fig = plot_fields(xs_u, zs_u[1:end-1], to_plot)

to_plot = OrderedDict(
    ("err", "err") => err
)

fig = plot_fields(xs_u, zs_u[1:end-1], to_plot)


# log10(dudz) = log10(2A) + n * log10(eff_stress)
A = 1.658286764403978e-25
log_ref_stress = 6.0
log_ref_strain_rate = log10.(2A) .+ 3 * log_ref_stress
log_ref_strain_rate - 4 * log_ref_stress