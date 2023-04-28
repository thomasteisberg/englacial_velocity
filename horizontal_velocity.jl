using CairoMakie

using PyCall
scipy_interpolate = pyimport_conda("scipy.interpolate", "scipy")

function layers_from_age_depth(xs, zs, age, layer_ages; edge_effect_m=500)
    layers = Vector{Function}(undef, length(layer_ages))
    test = Vector{Any}(undef, length(layer_ages))

    for (layer_idx, a) in enumerate(layer_ages)
        age_nonan = age
        age_nonan[isnan.(age_nonan)] .= -Inf
        idxs = argmin(abs.(age .- a), dims=2)
        layer_z = [zs[idx[2]] for idx in idxs][:,1]

        weights = vec(1 ./ (abs.(age[idxs] .- a) .+ 1e-6))
        edge_effects_idx = argmin(abs.(xs .- edge_effect_m))
        weights[1:edge_effects_idx] .= 0

        tck = scipy_interpolate.splrep(xs, layer_z, w=weights)
        function l(x)
            res = scipy_interpolate.splev(x, tck)
            if length(res) == 1
                return res[1]
            else
                return res
            end
        end

        layers[layer_idx] = l

        test[layer_idx] = idxs
    end

    return layers, test
end

function advect_layer(u, w, xs, initial_layer, layer_ages)
    layer_z = Vector{Float64}(undef, length(xs))
    try
        layer_z = initial_layer(xs)
    catch
        layer_z = @. initial_layer(xs)
    end
    u0 = vcat(xs', layer_z')
    t0 = 0.0

    layers = Vector{PyObject}(undef, length(layer_ages))

    function layer_velocity!(dxz, xz, p, t)
        # xz[1,:] is x, xz[2,:] is z
        dxz[1,:] = @. u(xz[1,:], xz[2,:])
        dxz[2,:] = @. w(xz[1,:], xz[2,:])
    end

    prob = ODEProblem(layer_velocity!, u0, (0.0, 1.0))

    for (layer_idx, layer_t) in enumerate(layer_ages)
        Δt = layer_t - t0
        
        prob = remake(prob, u0=u0, tspan=(0.0, Δt))
        sol = solve(prob)
        interp = scipy_interpolate.interp1d(sol.u[end][1,:], sol.u[end][2,:], kind="linear", fill_value="extrapolate")
    
        layers[layer_idx] = interp
    
        t0 = layer_t
    
        layer_z = interp(xs)
        u0 = vcat(xs', layer_z')
    end

    return layers
end


# function advect_layers(u, w, xs, layers, delta_ts)
#     function layer_velocity!(dxy, xy, p, t)
#         # xy[1] is x, xy[2] is y
#         dxy[1] = u(xy[1], xy[2])
#         dxy[2] = w(xy[1], xy[2])
#     end

#     # Dummy ODE problem defining the problem structure. We'll remake this problem
#     # with new initial conditions for each layer particle.
#     prob = ODEProblem(layer_velocity!, [0; 0], (0.0, 1.0))
#     layers_advected = Array{Any}(undef, length(layers_t0))
    
#     for (layer_idx, layer) in enumerate(layers)
#         function prob_func(prob, i, repeat)
#             x = 0#xs[i]
#             remake(prob, u0=[x; layer(x)], tspan=(0.0, delta_ts[layer_idx]))
#         end

#         layer_prob = EnsembleProblem(prob, prob_func=prob_func)
#         sol = solve(layer_prob, trajectories=length(xs))

#         transformed_layer_xs = [sol[i].u[end][1] for i = 1:length(xs)]
#         transformed_layer_zs = [sol[i].u[end][2] for i = 1:length(xs)]

#         # interp = scipy_interpolate.interp1d(transformed_layer_xs, transformed_layer_zs, kind="linear", fill_value="extrapolate")
#         # function l(x)
#         #     res = interp(x)
#         #     if length(x) == 1
#         #         return res[1]
#         #     else
#         #         return res
#         #     end
#         # end

#         # layers_advected[layer_idx] = l
#         #layers_advected[layer_idx] = linear_interpolation(transformed_layer_xs, transformed_layer_zs, extrapolation_bc=Line())
#     end

#     return layers_advected
# end

function horizontal_velocity(spatial_parameters::Tuple{Num, Num},
    d2l_dtdz::Function, d2l_dxdz::Function, dl_dx::Function;
    dx::Float64 = 750.0, dz::Float64 = 50.0)
    # PDE we want to solve:
    # d2l_dtdz + (u * d2l_dxdz) + (du_dz * dl_dx) + du_dx = 0
    #
    # Because u(x, z) is an already defined expression (representing "ground truth"),
    # we'll call the thing we're estimating u_est(x, z)
    # Rewritten:
    # d2l_dtdz + (u_est * d2l_dxdz) + (Dz(u_est) * dl_dx) + Dx(u_est) ~ 0

    x, z = spatial_parameters

    @variables u_est(..)

    # Spatial first derivative operators
    Dx = Differential(x)
    Dz = Differential(z)

    # Our PDE
    eq = [d2l_dtdz(x, z) + (u_est(x, z) * d2l_dxdz(x, z)) + (Dz(u_est(x, z)) * dl_dx(x, z)) + Dx(u_est(x, z)) ~ 0]

    # Boundary conditions
    bcs = [u_est(0, z) ~ u(0, z), # Horizontal velocity at x=0 -- pretty much need this
           u_est(x, domain_z) ~ u(x, domain_z)] # Horizontal velocity along the surface -- less necessary -- inteesting to play with this

    # Domain must be rectangular. Defined based on prior parameters
    domains = [x ∈ Interval(0.0, domain_x),
               z ∈ Interval(0.0, domain_z)]

    # x, z are independent variables. Solving for u_est(x, z)
    @named pdesys = PDESystem(eq, bcs, domains, [x, z], [u_est(x, z)])

    discretization = MOLFiniteDifference([x => dx, z => dz], nothing, approx_order=2)

    prob = discretize(pdesys, discretization, progress=true)

    # TODO: Think about appropriate solver algorithm. Auto-selection picks different options.
    sol = solve(prob)
    
    return sol
end

function plot_horizontal_velocity_result(x, z, sol, layers, u::Function)
    # Visualize result and compare with ground truth

    u_sol = sol[u_est(x, z)] # solver result on discretized grid

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
    h = heatmap!(ax, sol[x], sol[z], u_sol - u_true, colorrange=(-2, 2), colormap=:RdBu_5)
    Colorbar(fig[3, 2], h, label="Horizontal Velocity Difference [m/yr]")
    ylims!(ax, minimum(sol[z]), maximum(sol[z]))

    # Show layers for reference
    for (l_t0, l_t1) in zip(layers_t0, layers_t1)
        lines!(ax, xs, l_t0(xs), color=:gray, linestyle=:dash)
    end

    fig
end