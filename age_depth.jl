using MethodOfLines
using DomainSets
using NonlinearSolve

using PyCall
scipy_interpolate = pyimport_conda("scipy.interpolate", "scipy")

function age_depth(spatial_parameters::Tuple{Num, Num}, u, w,
    domain_x::Float64, domain_z::Float64)

    println("This function is just for comparison. You should probably use age_depth_curvilinear")

    seconds_per_year = 60 * 60 * 24 * 365.0 # m/s to m/yr conversion

    x, z = spatial_parameters

    @variables age(..)

    # Spatial first derivative operators
    Dx = Differential(x)
    Dz = Differential(z)

    # Age depth equation in steady state (DA/dt=0)
    eq = [seconds_per_year * u(x,z) * Dx(age(x, z)) + seconds_per_year * w(x, z) * Dz(age(x, z)) ~ 1.0]

    # Boundary condition -- surface is age 0
    # Note: does not use surface contour -- use age_depth_curvilinear for that
    bcs = [age(x, domain_z) ~ 0]

    # Domain must be rectangular. Defined based on prior parameters
    domains = [x ∈ Interval(0.0, domain_x),
               z ∈ Interval(0.0, domain_z)]

    # x, z are independent variables. Solving for u_est(x, z)
    @named pdesys = PDESystem(eq, bcs, domains, [x, z], [age(x, z)])

    # Discretization step size
    # Note: These MUST be floats. Easiest thing is just to add a ".0" :)
    #fd_dx, fd_dz = 2000.0, 200.0
    fd_dx = 1000.0
    fd_dz = 100.0

    discretization = MOLFiniteDifference([x => fd_dx, z => fd_dz], nothing)

    prob = discretize(pdesys, discretization, progress=true)
    sol = solve(prob, NewtonRaphson())

    age_sol = sol[age(x, z)] # solver result on discretized grid
    
    return sol[x], sol[z], age_sol
end

function age_depth_curvilinear(spatial_parameters::Tuple{Num, Num}, u, w,
    domain_x::Float64, surface::Function, dsdx::Function;
    interpolate_to_xz::Bool = true, fd_dq::Float64 = 0.1, fd_dp::Float64 = 1000.0,
    output_dx::Float64 = 500.0, output_dz::Float64 = 50.0)

    seconds_per_year = 60 * 60 * 24 * 365.0 # m/s to m/yr conversion

    x, z = spatial_parameters

    @parameters p q # Curvilinear grid parameters
    # p = x
    # q = z/surface(x)
    @variables age(..)

    # Spatial first derivative operators
    Dp = Differential(p)
    Dq = Differential(q)

    # Age depth equation in steady state (DA/dt=0)
    eq = [u(p, q * surface(p)) * (Dp(age(p, q)) - Dq(age(p, q)) * (q * surface(p) * dsdx(p) * (surface(p))^-2)) + w(p, q * surface(p)) * Dq(age(p, q)) * (surface(p))^-1 ~ 1.0 / seconds_per_year]

    # Boundary condition -- surface is age 0
    bcs = [age(p, 1.0) ~ 0]

    # Domain must be rectangular. Defined based on prior parameters
    domains = [p ∈ Interval(0.0, domain_x),
               q ∈ Interval(0.0, 1.0)]

    # x, z are independent variables. Solving for u_est(x, z)
    @named pdesys = PDESystem(eq, bcs, domains, [p, q], [age(p, q)])

    #fd_dq = 0.1
    #fd_dp = 1000.0 # TODO WTF

    discretization = MOLFiniteDifference([p => fd_dp, q => fd_dq], nothing)

    prob = discretize(pdesys, discretization, progress=true)
    sol = solve(prob, NewtonRaphson())

    age_sol = sol[age(p, q)] # solver result on discretized grid

    if interpolate_to_xz
        X = ones(length(sol[q]))' .* sol[p]
        Q = sol[q]' .* ones(length(sol[p]))
        Z = Q .* (@. surface(X))

        x_out = 0:output_dx:domain_x
        z_out = 0:output_dz:maximum(Z)
        X_out = ones(length(z_out))' .* x_out
        Z_out = z_out' .* ones(length(x_out))

        age_grid = scipy_interpolate.griddata((vec(X), vec(Z)),
                        vec(sol[age(p, q)]),
                        (X_out, Z_out)) # TODO
        return x_out, z_out, age_grid
    else # No interpolation, just return data in (p, q) coordinates
        return sol
    end
end

function plot_age(xs, zs, age; contour::Bool = true, colorrange=(0,10000))
    fig = Figure(resolution=(1000, 300))

    ax = Axis(fig[1, 1], title="Age")
    if contour
        h = contour!(ax, xs, zs, age,
                        levels=0:200:10000,
                        colorrange=colorrange
                    )
    else
        h = heatmap!(ax, xs, zs, age, colorrange=colorrange)
    end
    cb = Colorbar(fig[1, 2], h, label="Years")
    
    fig
end