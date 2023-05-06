using CairoMakie

using PyCall
scipy_interpolate = pyimport_conda("scipy.interpolate", "scipy")

using SciMLNLSolve

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

function estimate_layer_deformation(u::Function, w::Function, xs, layers_t0)
    layers_t1 = Vector{Union{Function, PyObject}}(undef, length(layers_t0))

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
                                    bbox=(0, domain_x, 0, domain_z),
                                    w=min.(1 ./ abs.(deformation_layer_slope), 1/0.05)
                                    )
    dl_dx_scipy_fn(x, z) = dl_dx_scipy(x, z)[1] # Julia function wrapper
    d2l_dxdz(x, z) = dl_dx_scipy.partial_derivative(0,1)(x, z)[1]

    debugging_dict = Dict(
        "deformation_xs" => deformation_xs,
        "deformation_zs" => deformation_zs,
        "deformation_delta_l" => deformation_delta_l,
        "deformation_layer_slope" => deformation_layer_slope,
        "dl_dt_scipy" => dl_dt_scipy,
        "dl_dx_scipy" => dl_dx_scipy,
        "dl_dt_scipy_fn" => dl_dt_scipy_fn,
        "dl_dx_scipy_fn" => dl_dx_scipy_fn,
        "d2l_dtdz" => d2l_dtdz,
        "d2l_dxdz" => d2l_dxdz
    )

    return (dl_dt_scipy_fn, d2l_dtdz, dl_dx_scipy_fn, d2l_dxdz, debugging_dict)
end

function estimate_layer_deformation_monotone(u::Function, w::Function, xs, zs, layers_t0)
    layers_t1 = Vector{Union{Function, PyObject}}(undef, length(layers_t0))

    for i in 1:length(layer_ages)
        layers_t1[i] = advect_layer(u, w, xs, layers_t0[i], 1.0*seconds_per_year)[1]
    end

    if xs[1] == 0 # x=0 is problematic for interpolation because all of the layers are at z=domain_z
        xs = xs[2:end]
    end

    dl_dt_grid = zeros(length(xs), length(zs))
    d2l_dtdz_grid = zeros(length(xs), length(zs))
    dl_dx_grid = zeros(length(xs), length(zs))
    d2l_dxdz_grid = zeros(length(xs), length(zs))


    # Monotone cubic spline interpolation to grid data
    for (x_idx, x) in enumerate(xs)
        layers_z = [(layers_t1[i](x)[1] + layers_t0[i](x)[1]) / 2 for i in eachindex(layers_t0)]
        layers_delta_l = [layers_t1[i](x)[1] - layers_t0[i](x)[1] for i in eachindex(layers_t0)]
        layers_slope = [(layers_t1[i](x + 1.0)[1] - layers_t1[i](x)[1] + layers_t0[i](x + 1.0)[1] - layers_t0[i](x)[1]) / 2 for i in eachindex(layers_t0)]

        dl_dt_interp = scipy_interpolate.PchipInterpolator(layers_z[end:-1:1], layers_delta_l[end:-1:1], extrapolate=true)
        dl_dt_grid[x_idx, :] = dl_dt_interp(zs)
        d2l_dtdz_grid[x_idx, :] = dl_dt_interp.derivative(1)(zs)

        dl_dx_interp = scipy_interpolate.PchipInterpolator(layers_z[end:-1:1], layers_slope[end:-1:1], extrapolate=true)
        dl_dx_grid[x_idx, :] = dl_dx_interp(zs)
        d2l_dxdz_grid[x_idx, :] = dl_dx_interp.derivative(1)(zs)
    end

    # Followed by linear interpolation to expand to 2D
    #X = xs .* ones(length(zs), 1)'
    #Z = ones(length(xs), 1) .* zs'
    #print(size(X))
    #print(size(Z))
    #print(size(dl_dt_grid))
    dl_dt_scipy = scipy_interpolate.RegularGridInterpolator((xs, zs), dl_dt_grid, fill_value=nothing, bounds_error=false)
    dl_dt_scipy_fn(x, z) = dl_dt_scipy([x, z])[1] # Julia function wrapper
    d2l_dtdz_scipy = scipy_interpolate.RegularGridInterpolator((xs, zs), d2l_dtdz_grid, fill_value=nothing, bounds_error=false)
    d2l_dtdz(x, z) = d2l_dtdz_scipy([x, z])[1]

    dl_dx_scipy = scipy_interpolate.RegularGridInterpolator((xs, zs), dl_dx_grid, fill_value=nothing, bounds_error=false)
    dl_dx_scipy_fn(x, z) = dl_dx_scipy([x, z])[1] # Julia function wrapper
    d2l_dxdz_scipy = scipy_interpolate.RegularGridInterpolator((xs, zs), d2l_dxdz_grid, fill_value=nothing, bounds_error=false)
    d2l_dxdz(x, z) = d2l_dxdz_scipy([x, z])[1]

    # Only included for debugging comparison
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

    debugging_dict = Dict(
        "dl_dt_grid" => dl_dt_grid,
        "d2l_dtdz_grid" => d2l_dtdz_grid,
        "dl_dx_grid" => dl_dx_grid,
        "d2l_dxdz_grid" => d2l_dxdz_grid,
        "deformation_xs" => deformation_xs,
        "deformation_zs" => deformation_zs,
        "deformation_delta_l" => deformation_delta_l,
        "deformation_layer_slope" => deformation_layer_slope,
    )

    #return debugging_dict

    return (dl_dt_scipy_fn, d2l_dtdz, dl_dx_scipy_fn, d2l_dxdz, debugging_dict)
end

function horizontal_velocity(spatial_parameters::Tuple{Num, Num},
    u::Function,
    d2l_dtdz::Function, d2l_dxdz::Function, dl_dx::Function;
    dx::Float64 = 500.0, dz::Float64 = 50.0)
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
    bcs = [u_est(0, z) ~ u(0, z) * seconds_per_year, # Horizontal velocity at x=0 -- pretty much need this
           u_est(x, domain_z) ~ u(x, domain_z) * seconds_per_year] # Horizontal velocity along the surface -- less necessary -- inteesting to play with this

    # Domain must be rectangular. Defined based on prior parameters
    domains = [x ∈ Interval(0.0, domain_x),
               z ∈ Interval(0.0, domain_z)]

    # x, z are independent variables. Solving for u_est(x, z)
    @named pdesys = PDESystem(eq, bcs, domains, [x, z], [u_est(x, z)])

    discretization = MOLFiniteDifference([x => dx, z => dz], nothing, approx_order=2)

    prob = discretize(pdesys, discretization, progress=true)

    sol = solve(prob, NewtonRaphson())
    
    return sol[x], sol[z], sol[u_est(x, z)]
end

function horizontal_velocity_curvilinear(spatial_parameters::Tuple{Num, Num},
    surface_velocity, inflow_velocity, surface, dsdx,
    d2l_dtdz::Function, d2l_dxdz::Function, dl_dx::Function;
    dp::Float64 = 400.0, dq::Float64 = 25.0/1200.0,
    interpolate_to_xz::Bool = true, output_dx::Float64 = 200.0, output_dz::Float64 = 50.0,
    u_true::Function)
    # PDE we want to solve:
    # d2l_dtdz + (u * d2l_dxdz) + (du_dz * dl_dx) + du_dx = 0
    #
    # Because u(x, z) is an already defined expression (representing "ground truth"),
    # we'll call the thing we're estimating u_est(x, z)
    # Rewritten:
    # d2l_dtdz + (u_est * d2l_dxdz) + (Dz(u_est) * dl_dx) + Dx(u_est) ~ 0

    x, z = spatial_parameters

    @parameters p q
    @variables u_est(..)

    # Spatial first derivative operators
    Dp = Differential(p)
    Dq = Differential(q)

    # Our PDE
    eq = [d2l_dtdz(p, q * surface(p)) + (u_est(p, q) * d2l_dxdz(p, q * surface(p))) + ((Dq(u_est(p, q)) / surface(p)) * dl_dx(p, q * surface(p))) + (Dp(u_est(p,q)) - Dq(u_est(p,q)) * q * dsdx(p) / surface(p)) ~ 0]
    # Boundary conditions
    bcs = [u_est(0, q) ~ inflow_velocity(q * surface(p)) * seconds_per_year,
           #u_est(p, 1.0) ~ surface_velocity(p) * seconds_per_year, # Horizontal velocity along the surface -- less necessary -- inteesting to play with this
           u_est(p, 0.0) ~ 0.0] # TODO: This shouldn't be necessary. Something is wrong with curvilinear case

    # Domain must be rectangular. Defined based on prior parameters
    domains = [p ∈ Interval(0.0, domain_x),
               q ∈ Interval(0.0, 1.0)]

    # x, z are independent variables. Solving for u_est(x, z)
    @named pdesys = PDESystem(eq, bcs, domains, [p, q], [u_est(p, q)])

    discretization = MOLFiniteDifference([p => dp, q => dq], nothing, approx_order=2)

    prob = discretize(pdesys, discretization, progress=true)
    println(typeof(prob))

    sol = MethodOfLines.solve(prob, NLSolveJL())#, NewtonRaphson(diff_type=Val{:central}))

    if interpolate_to_xz
        X = ones(length(sol[q]))' .* sol[p]
        Q = sol[q]' .* ones(length(sol[p]))
        Z = Q .* (@. surface(X))

        x_out = 0:output_dx:domain_x
        z_out = 0:output_dz:maximum(Z)
        X_out = ones(length(z_out))' .* x_out
        Z_out = z_out' .* ones(length(x_out))

        u_est_grid = scipy_interpolate.griddata((vec(X), vec(Z)),
                        vec(sol[u_est(p, q)]),
                        (X_out, Z_out)) # TODO
        return x_out, z_out, u_est_grid
    else # No interpolation, just return data in (p, q) coordinates
        return sol
    end
end

function plot_horizontal_velocity_result(xs, zs, u_sol, layers, u_true_fn::Function)
    # Visualize result and compare with ground truth

    u_true = (@. u_true_fn(xs, zs')) * seconds_per_year # Ground truth on PDE solution grid
    #clims = (0, max(maximum(u_sol[@. ~isnan.(u_sol)]), maximum(u_true)))
    clims = (0, maximum(u_true))
    #clims = (0, 10)

    fig = Figure(resolution=(1000, 1000))
    ax = Axis(fig[1, 1], title="PDE solution")
    h = heatmap!(ax, xs, zs, u_sol, colorrange=clims)
    Colorbar(fig[1, 2], h, label="Horizontal Velocity [m/yr]")

    # Ground truth for comparison
    ax = Axis(fig[2, 1], title="True values")
    h = heatmap!(ax, xs, zs, u_true, colorrange=clims)
    Colorbar(fig[2, 2], h, label="Horizontal Velocity [m/yr]")

    # Comparison between the two
    ax = Axis(fig[3, 1], title="solution - true values\n(layers shown as gray lines for reference)")
    h = heatmap!(ax, xs, zs, u_sol - u_true, colorrange=(-2, 2), colormap=:RdBu_5)
    Colorbar(fig[3, 2], h, label="Horizontal Velocity Difference [m/yr]")
    ylims!(ax, minimum(zs), maximum(zs))

    # Show layers for reference
    for l in layers
        lines!(ax, xs, l(xs), color=:gray, linestyle=:dash)
    end

    fig
end