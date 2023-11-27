using Revise

using CairoMakie
using Colors
using Symbolics
using Dates
using JLD2

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

println("Outputs will be saved with prefix: $timestr")

#  ============================
## Problem definition and setup
#  ============================

# Domain size
domain_x = 100000.0 # meters
domain_z = 3000.0 # meters

# Grids for when discretization is needed
dx = 100.0
dz = 25.0
xs = 0.0:dx:domain_x
zs = 0.0:dz:domain_z

# x, z are our spatial coordinates
# These will be used in various symbolic math throughout
@parameters x z

# Define surface geometry
surface(x) = domain_z - ((x / 18000.0)^3.0)

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

n = 2.5

println("n = $n")

reference_n = 2.5
u_ref, _, _ = sia_model((x, z), surface, dsdx; n=reference_n)

u_test, _, _ = sia_model((x, z), surface, dsdx, n=n)
basal_velocity(x) = u_ref(x, surface(x)) - u_test(x, surface(x))

u, w, dudx = sia_model((x, z), surface, dsdx; n=n, basal_velocity=basal_velocity)

function mask_above_surface(f, xs, zs)
    values = @. f(xs, zs')
    Z = zs' .* ones(length(xs))
    values[Z .> surface.(xs)] .= NaN
    return values
end

to_plot = OrderedDict(
        ("u (Horizontal Velocity)", "u [m/a]") => mask_above_surface(u, xs, zs) * seconds_per_year,
        #("u (Horizontal Velocity)", "u [m/a]") => (@. u(xs, zs')) * seconds_per_year,
        ("w (Vertical Velocity)", "w [m/a]") => mask_above_surface(w, xs, zs) * seconds_per_year,
        #("du/dx", "du/dx [a^-1]") => mask_above_surface(dudx, xs, zs) * seconds_per_year,
    )
fig = plot_fields(xs, zs, to_plot; surface=surface)

# w needs to be registered to be used in a PDESystem equation
# But it can't be registered after it's returned by a function, apparently
# 
# Approach 1: (kind of silly but simple workaround)
w_reg(x, z) = w(x, z)
@register w_reg(x, z)

save_figure(fig, "uw_sia")
fig

#  ================
## Surface Velocity
#  ================

begin
    fig = Figure(resolution=(1000, 500), fontsize=32)
    ax = Axis(fig[1, 1], title="Surface Horizontal Velocity", ylabel="Horizontal Velocity [m/yr]", xlabel="x [km]")
    lines!(ax, xs ./ 1000, seconds_per_year * @. u(xs, surface(xs)))

    fig
end

#  ===
## Alternative layers based on particle flow
#  ===

#layer_ages = 0:100:5000
layer_ages = 10 .^ (range(0,stop=4,length=20))
layers_t0 = advect_layer(u, w, xs, surface, layer_ages*seconds_per_year)

layers_t1 = Vector{Function}(undef, length(layer_ages))
for i in 1:length(layer_ages)
    layers_t1[i] = advect_layer(u, w, xs, layers_t0[i], 1.0*seconds_per_year)[1]
    #layers_t1[i] = add_measurement_noise_to_layer(layers_t1[i], 0.1, xs)
end

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

#  ====================
## Layer deformation functions
#  ====================

function layer_dl_dx(x, layer_idx, dx=1.0)
    # Numerical central difference approximation of layer slope
    # Output units are m/m
    (layers_t0[layer_idx](x .+ dx) .- layers_t0[layer_idx](x .- dx)) / (2 * dx)
end

function layer_dl_dt(x, layer_idx)
    # Numerical forward difference approximation of layer vertical deformation
    # Output units are m/year
    layers_t1[layer_idx](x) .- layers_t0[layer_idx](x) # dt = 1 year
end

function layer_d2l_dxdz(x, layer_idx)
    # Numerical central difference approximation of d^2l/(dxdz)
    layer_p1 = layer_dl_dx(x, layer_idx+1)
    layer_m1 = layer_dl_dx(x, layer_idx-1)
    layer_dz = layers_t0[layer_idx+1](x) .- layers_t0[layer_idx-1](x)
    (layer_p1 - layer_m1) ./ (layer_dz)
end

function layer_d2l_dtdz(x, layer_idx)
    # Numerical central difference approximation of d^2l/(dtdz)
    layer_p1 = layer_dl_dt(x, layer_idx+1)
    layer_m1 = layer_dl_dt(x, layer_idx-1)
    layer_dz = layers_t0[layer_idx+1](x) .- layers_t0[layer_idx-1](x)
    (layer_p1 - layer_m1) ./ (layer_dz)
end

#  =========================
## Method Of Characteristics
#  =========================

function dv_ds(v, p, s)
    return -1 * layer_d2l_dxdz(s, p)[1]*v - layer_d2l_dtdz(s, p)[1]
end

start_pos_x = 10
prob = ODEProblem(dv_ds, 0, (start_pos_x, domain_x), 1)

layer_sols = Vector{ODESolution}(undef, length(layers_t0))

for p in 2:length(layers_t0)-1
    z0 = layers_t0[p](0)[1]
    v0 = seconds_per_year * u(start_pos_x, z0) # TODO offset?
    global prob = remake(prob, p=p, u0=v0)
    layer_sols[p] = solve(prob)
    println(layer_sols[p].retcode)
end

##

begin
    fig = Figure(resolution=(1000, 1000), fontsize=32)

    # Solution
    ax = Axis(fig[1, 1], title="Horizontal Velocity Solution", xlabel="x [km]", ylabel="z [m]")
    crange = (0, 50)

    #h = heatmap!(ax, xs ./ 1000, zs, (@. u(xs, zs')) .* seconds_per_year, colorrange=crange)
    #cb = Colorbar(fig[1,2], h, label="u [m/yr]")

    sc = nothing
    for p in 2:length(layers_t0)-1
        sc = plot!(ax, layer_sols[p].t ./ 1000, layers_t0[p](layer_sols[p].t), color=layer_sols[p].u, colorrange=crange)
    end
    cb = Colorbar(fig[1,2], sc, label="u [m/yr]")

    # Error
    ax = Axis(fig[2, 1], title="Error (predicted - true)", xlabel="x [km]", ylabel="z [m]")
    crange = (-1, 1)
    sc = nothing

    for p in 2:length(layers_t0)-1
        xs_l = layer_sols[p].t
        zs_l = layers_t0[p](layer_sols[p].t)
        u_true = (@. u(xs_l, zs_l)) .* seconds_per_year
        global sc = plot!(ax, xs_l ./ 1000, zs_l, color=layer_sols[p].u .- u_true, colorrange=crange, colormap=:RdBu_5)
    end
    cb = Colorbar(fig[2,2], sc, label="Error [m/yr]")

    # # Error
    # ax = Axis(fig[3, 1], title="Error % (predicted - true)/true", xlabel="x [km]", ylabel="z [m]")
    # crange = (-10, 10)
    # sc = nothing

    # for p in 2:length(layers_t0)-1
    #     xs_l = layer_sols[p].t
    #     zs_l = layers_t0[p](layer_sols[p].t)
    #     u_true = (@. u(xs_l, zs_l)) .* seconds_per_year
    #     global sc = plot!(ax, xs_l ./ 1000, zs_l, color=100 * (layer_sols[p].u .- u_true) ./ u_true, colorrange=crange, colormap=:RdBu_5)
    # end
    # cb = Colorbar(fig[3,2], sc, label="Error %")

    save_figure(fig, "results")
    fig
end

#  =============================
## Quality of layer estimate based on ODE stability
#  =============================

# Iterate over each layer and find the percentage of points where d2l_dxdz is negative
layer_stability = NaN * zeros(length(layers_t0))
for layer_idx = 2:length(layers_t0)-1
    b_layer = layer_d2l_dxdz(xs, layer_idx)
    layer_stability[layer_idx] = sum(b_layer .> 0.0) / length(b_layer)
    println("Layer ", layer_idx, " stability: ", layer_stability[layer_idx])
end

#  =============================
## Ice effective viscosity from finite difference of MOC solution lines
#  =============================

ρ = 918 # kg/m^3
g = 9.81 # m/s^2
x_offset_left, x_offset_right = 8000, 1000
z_offset_bottom, z_offset_top = 300, 300
minimum_z_spacing_visc = 10

# Find du/dz from the finite differences along the layer_sols lines
x_visc = []
z_visc = []
dudz_visc = []
eff_stress_visc = []
layer_idxs = []
stability = []
delta_zs = []

minimum_z_spacing_visc = 400

for x_pos = x_offset_left:100:(domain_x-x_offset_right)
    last_z = -Inf
    for layer_idx = 2:1:length(layers_t0)-2
        z_pos = layers_t0[layer_idx](x_pos)[1]

        delta_z = (layers_t0[layer_idx-1](x_pos)[1] - layers_t0[layer_idx+1](x_pos)[1])
        if delta_z < minimum_z_spacing_visc
            continue
        end
        if x_pos % 10000 == 0
            println("Layer Idx: $layer_idx, delta_z: $delta_z, x_pos: $x_pos")
        end
        if (abs(z_pos - last_z) > minimum_z_spacing_visc) && (z_pos >= z_offset_bottom) && (z_pos <= domain_z - z_offset_top)
            # Estimate du_dz
            dudz_central_diff = (layer_sols[layer_idx-1](x_pos) - layer_sols[layer_idx+1](x_pos)) / ( seconds_per_year * (layers_t0[layer_idx-1](x_pos)[1] - layers_t0[layer_idx+1](x_pos)[1]))
            
            # Estimate effective stress under SIA
            dist_to_surf = surface(x_pos) - z_pos
            rheology_eff_stress = ρ * g * dist_to_surf * -dsdx(x_pos) # (3.88)

            push!(x_visc, x_pos)
            push!(z_visc, z_pos)
            push!(delta_zs, delta_z)
            push!(dudz_visc, dudz_central_diff)
            push!(eff_stress_visc, rheology_eff_stress)
            push!(layer_idxs, layer_idx)
            push!(stability, layer_stability[layer_idx])

            last_z = z_pos
        end
    end
end

dudz_visc[dudz_visc .< 0] .= NaN

begin
    fig = Figure(resolution=(1000, 1000), fontsize=32)
    ax = Axis(fig[1, 1], title="Effective stress vs strain rate", xlabel="log(stress)", ylabel="log(strain rate)")
    s = scatter!(ax, log10.(eff_stress_visc), log10.(dudz_visc), markersize=4) #, color=Float32.(delta_zs))
    #cb = Colorbar(fig[1,2], s, label="Layer stability")
    stress_xs = 0:0.1:5.5
    A_n2p5 = 1.6582867644039821e-22
    A_n3p5 = 1.658286764403982e-28

    lines!(ax, stress_xs, 2.5 .* stress_xs .+ log10(2*A_n2p5), linestyle=:dash, color=:black)
    lines!(ax, stress_xs, 3 .* stress_xs .+ log10(2*A_n3), linestyle=:dash, color=:black)
    lines!(ax, stress_xs, 3.5 .* stress_xs .+ log10(2*A_n3p5), linestyle=:dash, color=:black)

    # A_n3 = 1.658286764403978e-25
    # lines!(ax, stress_xs, 3 .* stress_xs .+ log10(2*A_n3), linestyle=:dash, color=:black)
    # A_n4 = 1.658286764403982e-31
    # lines!(ax, stress_xs, 4 .* stress_xs .+ log10(2*A_n4), linestyle=:dash, color=:black)
    # A_n2 = 1.658286764403982e-19
    # lines!(ax, stress_xs, 2 .* stress_xs .+ log10(2*A_n2), linestyle=:dash, color=:black)
    xlims!(3.5, 5.5)
    ylims!(-13,-8)
    fig
end

save_figure(fig, "rheology_moc")
fig

#  =============================
## Stress/strain at a single position from each layer
#  =============================

ρ = 918 # kg/m^3
g = 9.81 # m/s^2
minimum_z_spacing_visc = 10

x_pos = 95e3
x_idx = argmin(abs.(xs .- x_pos))

# Iterate over each layer and find the percentage of points where d2l_dxdz is negative
layer_stability_to_point = zeros(length(layers_t0))
for layer_idx = 2:length(layers_t0)-1
    xs_tmp = 0.0:10.0:x_pos
    b_layer = layer_d2l_dxdz(xs_tmp, layer_idx)
    layer_stability_to_point[layer_idx] = sum(b_layer .> 0.0) / length(b_layer)
    println("Layer ", layer_idx, " stability % up to ", (x_pos/1e3), " km: ", layer_stability_to_point[layer_idx])
end

# Find du/dz from the finite differences along the layer_sols lines
x_visc = Vector{Float32}()
z_visc = Vector{Float32}()
dudz_visc = Vector{Float32}()
u_visc = Vector{Float32}()
eff_stress_visc = Vector{Float32}()
layer_idxs = Vector{Float32}()
stability = Vector{Float32}()

last_z = -Inf
for layer_idx = 3:1:length(layers_t0)-2
    z_pos = layers_t0[layer_idx](x_pos)[1]
    if (abs(z_pos - last_z) > minimum_z_spacing_visc)# && (z_pos >= z_offset_bottom) && (z_pos <= domain_z - z_offset_top)
        # Estimate du_dz
        dudz_central_diff = (layer_sols[layer_idx-1](x_pos) - layer_sols[layer_idx+1](x_pos)) / ( seconds_per_year * (layers_t0[layer_idx-1](x_pos)[1] - layers_t0[layer_idx+1](x_pos)[1]))
        
        u_centeral_mean = (layer_sols[layer_idx-1](x_pos) + layer_sols[layer_idx+1](x_pos)) / 2
        
        # Estimate effective stress under SIA
        dist_to_surf = surface(x_pos) - z_pos
        rheology_eff_stress = ρ * g * dist_to_surf * -dsdx(x_pos) # (3.88)

        push!(x_visc, x_pos)
        push!(z_visc, z_pos)
        push!(dudz_visc, dudz_central_diff)
        push!(u_visc, u_centeral_mean)
        push!(eff_stress_visc, rheology_eff_stress)
        push!(layer_idxs, layer_idx)
        push!(stability, layer_stability_to_point[layer_idx])

        last_z = z_pos
    end
end

begin
    fig = Figure(resolution=(1000, 1000), fontsize=32)
    cmap = cgrad([:red, :red, :black], [0.0, 0.5, 1.0])
    
    zs_u = 0:10.0:surface(x_pos)
    u_true = seconds_per_year .* (@. u(x_pos, zs_u))
    dudz_fd_true = diff(u_true) ./ diff(zs_u)

    ax = Axis(fig[1, 1], title="Horizontal Velocity (n=$(n))", xlabel="u [m/yr]", ylabel="Elevation [m]")
    lines!(ax, u_true, zs_u, color=:black, linestyle=:dash, label="True")
    scatter!(ax, u_visc, z_visc, label="ODE Result")#, color=stability, colormap=cmap)
    ylims!(ax, 0, surface(x_pos))
    xlims!(ax, 0, 50)
    axislegend(ax, position=:lt)

    ax = Axis(fig[1, 2], title="dudz")
    lines!(ax, dudz_fd_true, zs_u[1:end-1], color=:black, linestyle=:dash)
    plot!(ax, seconds_per_year*dudz_visc, z_visc, color=stability, colormap=cmap)
    ylims!(ax, 0, surface(x_pos))

    # ax = Axis(fig[1, 3], title="Log effective stress")
    # lines!(ax, log10.(max.(rheology_eff_stress[x_idx, :],1e-12)), zs_u[1:end-1], color=:black, linestyle=:dash)
    # ylims!(ax, 0, surface(x_pos))
    # xlims!(ax, 3, 6)

    fig
end

save_figure(fig, "rheology_single_x")
fig

#  =============================
## Save key results for later plot generation
#  =============================

filename = "plots/$(timestr)-data.jld2"
jldsave(filename; domain_x, domain_z, dx, dz, xs, zs, surface, dsdx, n, u, w,
                    dudx, layers_t0, layers_t1, layer_sols, timestr)