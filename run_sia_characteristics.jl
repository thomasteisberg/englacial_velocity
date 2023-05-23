using Revise

using CairoMakie
using Colors
using Symbolics
using OrderedCollections
using Dates

ENV["PYTHON"] = ""
using PyCall
scipy_interpolate = pyimport_conda("scipy.interpolate", "scipy")

includet("sia.jl")
includet("age_depth.jl")
includet("horizontal_velocity.jl")
includet("plot_helpers.jl")

seconds_per_year = 60 * 60 * 24 * 365.0 # m/s to m/yr conversion

timestr = Dates.format(Dates.now(), dateformat"YYYYmmdd-HHMM")

function save_figure(fig, filename_tag)
    filename = "plots/$(timestr)-$(filename_tag).png"
    save(filename, fig)
    return fig, filename
end

#  ============================
## Problem definition and setup
#  ============================

# Domain size
domain_x = 15000.0 # meters
domain_z = 2000.0 # meters

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
save_figure(fig, "surface")
fig

#  ================================================
## Generate 2D (x, z) velocity field from SIA model
#  ================================================

u, w, dudx = sia_model((x, z), surface, dsdx; n=2.0)

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

save_figure(fig, "uw_sia")
fig

#  ===
## Alternative layers based on particle flow
#  ===

#layer_ages = 0:100:5000
layer_ages = 10 .^ (range(0,stop=3,length=20))
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

save_figure(fig, "layers")
fig

#  ==========================
## Estimate layer deformation
#  ==========================

dl_dt, d2l_dtdz, dl_dx, d2l_dxdz, deformation_debug = estimate_layer_deformation(u, w, xs, layers_t0, noise_std=0.0)

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
save_figure(fig, "inputs")
fig

#  =========================
## Method Of Characteristics
#  =========================

function dv_ds(v, p, s)
    return -1 * d2l_dxdz(s, layers_t0[p](s)[1])*v - d2l_dtdz(s, layers_t0[p](s)[1])
end
prob = ODEProblem(dv_ds, 0, (0.0, domain_x), 1, saveat=100, abstol=1e-12)

layer_sols = Vector{ODESolution}(undef, length(layers_t0))

for p in 1:length(layers_t0)
    z0 = layers_t0[p](0)[1]
    v0 = u(10, z0) # TODO offset?
    prob = remake(prob, p=p, u0=v0)
    layer_sols[p] = solve(prob)
    println(layer_sols[p].retcode)
end

begin
    fig = Figure(resolution=(1000, 600))

    # Solution
    ax = Axis(fig[1, 1], title="Solutions (points) overlaid on true values (basemap)")
    crange = (0, 200)

    h = heatmap!(ax, xs, zs, (@. u(xs, zs')) .* seconds_per_year, colorrange=crange)
    cb = Colorbar(fig[1,2], h, label="u [m/yr]")

    for p in 1:length(layers_t0)
        plot!(ax, layer_sols[p].t, layers_t0[p](layer_sols[p].t), color=layer_sols[p].u, colorrange=crange)
    end

    # Error
    ax = Axis(fig[2, 1], title="Error (predicted - true)")
    crange = (-10, 10)
    sc = nothing

    for p in 1:length(layers_t0)
        xs_l = layer_sols[p].t
        zs_l = layers_t0[p](layer_sols[p].t)
        u_true = (@. u(xs_l, zs_l)) .* seconds_per_year
        sc = plot!(ax, xs_l, zs_l, color=layer_sols[p].u .- u_true, colorrange=crange, colormap=:RdBu_5)
    end
    cb = Colorbar(fig[2,2], sc, label="Error [m/yr]")

    fig
end

xs_u, zs_u = xs, zs

xs_l = collect(Iterators.flatten(layer_sols[p].t for p in 1:length(layers_t0)))
zs_l = collect(Iterators.flatten(layers_t0[p](layer_sols[p].t) for p in 1:length(layers_t0)))
u_l = collect(Iterators.flatten(layer_sols[p].u for p in 1:length(layers_t0)))

X = xs_u .* ones(length(zs_u), 1)'
Z = ones(length(xs_u), 1) .* zs_u'

u_est = scipy_interpolate.griddata((xs_l, zs_l), u_l, (X, Z), method="cubic", fill_value=NaN)

fig = plot_horizontal_velocity_result(xs_u, zs_u, u_est, layers_t0, u)
save_figure(fig, "pderesult")
fig

#  =============================
## Ice effective viscosity from finite difference of MOC solution lines
#  =============================

ρ = 918 # kg/m^3
g = 9.81 # m/s^2
x_offset_left, x_offset_right = 1000, 1000
z_offset_bottom, z_offset_top = 200, 1000

# Find du/dz from the finite differences along the layer_sols lines
x_visc = []
z_visc = []
dudz_visc = []
eff_stress_visc = []
layer_idxs = []

minimum_z_spacing_visc = 50

for x_pos = x_offset_left:100:(domain_x-x_offset_right)
    last_z = -Inf
    for layer_idx = 2:1:length(layers_t0)-1
        z_pos = layers_t0[layer_idx](x_pos)[1]
        if (abs(z_pos - last_z) > minimum_z_spacing_visc) && (z_pos >= z_offset_bottom) && (z_pos <= domain_z - z_offset_top)
            # Estimate du_dz
            dudz_central_diff = (layer_sols[layer_idx-1](x_pos) - layer_sols[layer_idx+1](x_pos)) / ( seconds_per_year * (layers_t0[layer_idx-1](x_pos)[1] - layers_t0[layer_idx+1](x_pos)[1]))
            
            # Estimate effective stress under SIA
            dist_to_surf = surface(x_pos) - z_pos
            rheology_eff_stress = ρ * g * dist_to_surf * -dsdx(x_pos) # (3.88)

            push!(x_visc, x_pos)
            push!(z_visc, z_pos)
            push!(dudz_visc, dudz_central_diff)
            push!(eff_stress_visc, rheology_eff_stress)
            push!(layer_idxs, layer_idx)

            last_z = z_pos
        end
    end
end

dudz_visc[dudz_visc .< 0] .= NaN

begin
    fig = Figure(resolution=(1000, 1000))
    ax = Axis(fig[1, 1], title="Effective stress vs strain rate (MOL FD)")
    scatter!(ax, log10.(eff_stress_visc), log10.(dudz_visc), markersize=4, color=Float32.(layer_idxs))
    stress_xs = 0:0.1:5.5
    A_n3 = 1.658286764403978e-25
    lines!(ax, stress_xs, 3 .* stress_xs .+ log10(2*A_n3), linestyle=:dash, color=:black)
    A_n4 = 1.658286764403982e-31
    lines!(ax, stress_xs, 4 .* stress_xs .+ log10(2*A_n4), linestyle=:dash, color=:black)
    A_n2 = 1.658286764403982e-19
    lines!(ax, stress_xs, 2 .* stress_xs .+ log10(2*A_n2), linestyle=:dash, color=:black)
    xlims!(3, 6)
    ylims!(-15,-5)
    fig
end


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

valid_regions_mask = ((xs_u .> x_offset_left) .& (xs_u .< (domain_x - x_offset_right))) .* ((zs_u[1:end-1] .> z_offset_bottom) .& (zs_u[1:end-1] .< (domain_z-z_offset_top)))'
valid_regions_mask .&= .! isnan.(dudz_fd) # Filter out NaN values
valid_regions_mask .&= (dudz_fd .> 0.0) # Filter out negative values (TODO: slightly sus)

if minimum(dudz_fd[.! isnan.(dudz_fd)]) < -1e-10
    println("WARNING: dudz_fd has large negative values. This should not happen.")
    println("         Minimum value: ", minimum(dudz_fd[.! isnan.(dudz_fd)]))
end

rheology_eff_stress = (ρ * g * dist_to_surf_fd .* -(@. dsdx(xs_u))) # (3.88)
rheology_log_eff_stress = log10.( rheology_eff_stress[valid_regions_mask] )
rheology_log_eff_strain = log10.( (dudz_fd[valid_regions_mask]) )
#rheology_log_eff_strain = log10.( (dudz_fd_true[valid_regions_mask]) )

X_u = xs_u .* ones(length(zs_u), 1)'
Z_u = ones(length(xs_u), 1) .* zs_u'
X_u_valid = X_u[:, 1:end-1][valid_regions_mask]
Z_u_valid = Z_u[:, 1:end-1][valid_regions_mask]

begin
    fig = Figure(resolution=(1000, 1000))
    ax = Axis(fig[1, 1], title="Effective stress vs strain rate")
    scatter!(ax, rheology_log_eff_stress, rheology_log_eff_strain, markersize=2, color=Z_u_valid)
    stress_xs = 0:0.1:5.5
    #lines!(ax, stress_xs, 3 .* stress_xs .- 39.1, linestyle=:dash, color=:black)
    A_n3 = 1.658286764403978e-25
    lines!(ax, stress_xs, 3 .* stress_xs .+ log10(2*A_n3), linestyle=:dash, color=:black)
    A_n4 = 1.658286764403982e-31
    lines!(ax, stress_xs, 4 .* stress_xs .+ log10(2*A_n4), linestyle=:dash, color=:black)
    A_n2 = 1.658286764403982e-19
    lines!(ax, stress_xs, 2 .* stress_xs .+ log10(2*A_n2), linestyle=:dash, color=:black)
    xlims!(3, 6)
    ylims!(-15,-5)
    fig
end

save_figure(fig, "rheology")
fig

#  ====================
## dudz in valid region 
#  ====================

masked_dudz_fd = copy(dudz_fd) * seconds_per_year
masked_dudz_fd[.! valid_regions_mask] .= NaN
masked_dudz_fd_true = copy(dudz_fd_true) * seconds_per_year
masked_dudz_fd_true[.! valid_regions_mask] .= NaN

to_plot = OrderedDict(
        ("dudz_fd (estimated)", "dudz") => (masked_dudz_fd, Dict(:colorrange => (-0.1, 0.1), :colormap => :RdBu_5)),
        ("dudz_fd (true)", "dudz") => (masked_dudz_fd_true, Dict(:colorrange => (-0.1, 0.1), :colormap => :RdBu_5)),
        ("difference", "") => (masked_dudz_fd - masked_dudz_fd_true, Dict(:colorrange => (-0.05, 0.05), :colormap => :RdBu_5))
)

fig = plot_fields(xs_u, zs_u[1:end-1], to_plot)
save_figure(fig, "dudz_region")
fig

## Scratch space

1

# # test -- stress
# A = 1.658286764403978e-25
# x_pos, z_pos = 5000, 900
# eff_stress = -(dsdx(x_pos) * ρ * g * (surface(x_pos) - z_pos))
# x_idx, z_idx = argmin(abs.(xs_u .- x_pos)), argmin(abs.(zs_u .- z_pos))
# rheology_eff_stress[x_idx, z_idx]
# (rheology_eff_stress[x_idx, z_idx]) / eff_stress

# # test - strain rate
# dudz_test = 2 * A * eff_stress^3
# dudz_fd_true[x_idx, z_idx]
# dudz_fd_true[x_idx, z_idx] / dudz_test

# # test
# err = ((2 * A * rheology_eff_stress .^ 3) ./ dudz_fd_true)





# to_plot = OrderedDict(
#     ("err", "err") => err
# )

# fig = plot_fields(xs_u, zs_u[1:end-1], to_plot)


# # log10(dudz) = log10(2A) + n * log10(eff_stress)
# A = 1.658286764403978e-25
# log_ref_stress = 6.0
# log_ref_strain_rate = log10.(2A) .+ 3 * log_ref_stress
# log_ref_strain_rate - 4 * log_ref_stress