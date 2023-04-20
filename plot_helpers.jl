using CairoMakie
using OrderedCollections


function plot_fields(to_plot::OrderedDict)
    fig = Figure(resolution=(1000, 300*length(to_plot)))

    for (idx, titles) in enumerate(keys(to_plot))
        ax = Axis(fig[idx, 1], title=titles[1])
        h = heatmap!(ax, xs, zs, to_plot[titles])
        cb = Colorbar(fig[idx, 2], h, label=titles[2])
    end

    fig
end