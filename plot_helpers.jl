using CairoMakie
using OrderedCollections


function plot_fields(xs, zs, to_plot::OrderedDict; surface::Function=nothing)
    fig = Figure(resolution=(1000, 300*length(to_plot)))

    for (idx, titles) in enumerate(keys(to_plot))
        title, axis_label = titles

        ax = Axis(fig[idx, 1], title=title)

        if typeof(to_plot[titles]) <: Tuple
            h = heatmap!(ax, xs, zs, to_plot[titles][1]; to_plot[titles][2]...)
            cb = Colorbar(fig[idx, 2], h, label=axis_label)
        else
            h = heatmap!(ax, xs, zs, to_plot[titles])
            cb = Colorbar(fig[idx, 2], h, label=axis_label)
        end

        if surface != nothing
            lines!(ax, xs, (@. surface(xs)), linewidth=1, color=:black)
        end
    end

    fig
end