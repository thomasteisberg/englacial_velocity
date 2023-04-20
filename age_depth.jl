using MethodOfLines
using DomainSets
using NonlinearSolve

function testingtesting(spatial_parameters::Tuple{Num, Num}, u::Function, w::Function,
    domain_x::Float64, domain_z::Float64)
    return typeof(w)
end

function age_depth(spatial_parameters::Tuple{Num, Num}, u, w,
    domain_x::Float64, domain_z::Float64)

    seconds_per_year = 60 * 60 * 24 * 365.0 # m/s to m/yr conversion

    x, z = spatial_parameters

    @variables age(..)

    # Spatial first derivative operators
    Dx = Differential(x)
    Dz = Differential(z)

    # Age depth equation in steady state (DA/dt=0)
    eq = [seconds_per_year * u(x,z) * Dx(age(x, z)) + seconds_per_year * w(x, z) * Dz(age(x, z)) ~ 1.0]

    # Boundary condition -- surface is age 0
    # TODO: Use actual surface contour?
    bcs = [age(x, domain_z) ~ 0]

    # Domain must be rectangular. Defined based on prior parameters
    domains = [x ∈ Interval(0.0, domain_x),
               z ∈ Interval(0.0, domain_z)]

    # x, z are independent variables. Solving for u_est(x, z)
    @named pdesys = PDESystem(eq, bcs, domains, [x, z], [age(x, z)])

    # Discretization step size
    # Note: These MUST be floats. Easiest thing is just to add a ".0" :)
    #fd_dx, fd_dz = 2000.0, 200.0
    fd_dx = 500.0
    fd_dz = 50.0

    discretization = MOLFiniteDifference([x => fd_dx, z => fd_dz], nothing)

    prob = discretize(pdesys, discretization, progress=true)
    sol = solve(prob, NewtonRaphson())

    age_sol = sol[age(x, z)] # solver result on discretized grid
    
    return sol[x], sol[z], age_sol
end

function plot_age(xs, zs, age)
    fig = Figure(resolution=(1000, 300))

    ax = Axis(fig[1, 1], title="Age")
    h = contour!(ax, xs, zs, age,
                    levels=100:200:10000,
                    colorrange=(0,10000)
                )
    cb = Colorbar(fig[1, 2], h, label="Years")
    
    fig
end