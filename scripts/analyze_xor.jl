using RJBNN

function argument(name, default, convert=identity)
    prefix = "--$(name)="
    match = findfirst(value -> startswith(value, prefix), ARGS)
    match === nothing && return default
    return convert(split(ARGS[match], "="; limit=2)[2])
end

input_path = argument("input", nothing)
input_path === nothing && error("provide --input=results/xor/<result>.jls")
output_dir = argument("output-dir", joinpath(dirname(input_path), "analysis"))
burn_in_requested = argument("burn-in", 0, value -> parse(Int, value))

result = load_xor_result(input_path)
isempty(result.traces) && error("result contains no traces")
burn_in = clamp(burn_in_requested, 0, length(result.traces) - 1)
posterior_traces = result.traces[(burn_in + 1):end]
mkpath(output_dir)

function xml_escape(value)
    escaped = replace(string(value), "&" => "&amp;")
    escaped = replace(escaped, "<" => "&lt;")
    return replace(escaped, ">" => "&gt;")
end

function scale_values(values, low, high)
    minimum_value, maximum_value = extrema(values)
    if minimum_value == maximum_value
        return fill((low + high) / 2, length(values))
    end
    return low .+ (values .- minimum_value) .* ((high - low) / (maximum_value - minimum_value))
end

function write_line_plot(path, values; title, y_label, lower=nothing, upper=nothing)
    width, height = 800, 420
    left, right, top, bottom = 75, 25, 50, 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = length(values) == 1 ? [left + plot_width / 2] : collect(range(left, left + plot_width; length=length(values)))
    data_low = lower === nothing ? minimum(values) : lower
    data_high = upper === nothing ? maximum(values) : upper
    data_low == data_high && (data_high = data_low + 1)
    y_values = top .+ (data_high .- values) .* (plot_height / (data_high - data_low))
    points = join(["$(round(x; digits=2)),$(round(y; digits=2))" for (x, y) in zip(x_values, y_values)], " ")

    open(path, "w") do io
        print(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">
<rect width="100%" height="100%" fill="white"/>
<text x="$(width / 2)" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">$(xml_escape(title))</text>
<line x1="$left" y1="$top" x2="$left" y2="$(top + plot_height)" stroke="#333"/>
<line x1="$left" y1="$(top + plot_height)" x2="$(left + plot_width)" y2="$(top + plot_height)" stroke="#333"/>
<polyline fill="none" stroke="#1769aa" stroke-width="2" points="$points"/>
<text x="$(width / 2)" y="$(height - 15)" text-anchor="middle" font-family="sans-serif" font-size="14">Iteration</text>
<text x="18" y="$(height / 2)" text-anchor="middle" transform="rotate(-90 18 $(height / 2))" font-family="sans-serif" font-size="14">$(xml_escape(y_label))</text>
<text x="$(left - 8)" y="$(top + 5)" text-anchor="end" font-family="sans-serif" font-size="12">$(round(data_high; digits=3))</text>
<text x="$(left - 8)" y="$(top + plot_height)" text-anchor="end" font-family="sans-serif" font-size="12">$(round(data_low; digits=3))</text>
</svg>
""")
    end
end

function write_width_histogram(path, widths, maximum_width)
    counts = [count(==(width), widths) for width in 1:maximum_width]
    canvas_width, canvas_height = 800, 420
    left, right, top, bottom = 75, 25, 50, 60
    plot_width = canvas_width - left - right
    plot_height = canvas_height - top - bottom
    maximum_count = max(maximum(counts), 1)
    slot = plot_width / maximum_width

    open(path, "w") do io
        print(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$canvas_width" height="$canvas_height" viewBox="0 0 $canvas_width $canvas_height">
<rect width="100%" height="100%" fill="white"/>
<text x="$(canvas_width / 2)" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Observed hidden-width values</text>
<line x1="$left" y1="$top" x2="$left" y2="$(top + plot_height)" stroke="#333"/>
<line x1="$left" y1="$(top + plot_height)" x2="$(left + plot_width)" y2="$(top + plot_height)" stroke="#333"/>
""")
        for width in 1:maximum_width
            bar_height = counts[width] * plot_height / maximum_count
            x = left + (width - 1) * slot + 2
            y = top + plot_height - bar_height
            print(io, "<rect x=\"$x\" y=\"$y\" width=\"$(max(slot - 4, 1))\" height=\"$bar_height\" fill=\"#1769aa\"/>\n")
            print(io, "<text x=\"$(x + slot / 2 - 2)\" y=\"$(top + plot_height + 18)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\">$width</text>\n")
        end
        print(io, """<text x="$(canvas_width / 2)" y="$(canvas_height - 10)" text-anchor="middle" font-family="sans-serif" font-size="14">Hidden width k</text>
<text x="18" y="$(canvas_height / 2)" text-anchor="middle" transform="rotate(-90 18 $(canvas_height / 2))" font-family="sans-serif" font-size="14">Occurrences</text>
</svg>
""")
    end
end

function probability_color(probability)
    p = clamp(probability, 0, 1)
    red = round(Int, 230 * (1 - p) + 45 * p)
    green = round(Int, 70 * (1 - abs(2p - 1)) + 90)
    blue = round(Int, 55 * (1 - p) + 220 * p)
    return "rgb($red,$green,$blue)"
end

function write_decision_surface(path, traces, data; low=-1.0, high=1.0, resolution=50)
    axis = collect(range(low, high; length=resolution))
    grid = Matrix{Float64}(undef, resolution * resolution, 2)
    position = 1
    for y in reverse(axis), x in axis
        grid[position, :] = [x, y]
        position += 1
    end
    probabilities = zeros(Float64, size(grid, 1))
    for trace in traces
        probabilities .+= predict_probabilities(trace, grid)[1, :]
    end
    probabilities ./= length(traces)

    canvas = 650
    margin = 55
    plot_size = canvas - 2margin
    cell = plot_size / resolution
    coordinate(value) = margin + (value - low) * plot_size / (high - low)

    open(path, "w") do io
        print(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$canvas" height="$canvas" viewBox="0 0 $canvas $canvas">
<rect width="100%" height="100%" fill="white"/>
<text x="$(canvas / 2)" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Posterior-averaged XOR classification</text>
""")
        for index in eachindex(probabilities)
            row = (index - 1) ÷ resolution
            column = (index - 1) % resolution
            x = margin + column * cell
            y = margin + row * cell
            print(io, "<rect x=\"$x\" y=\"$y\" width=\"$(cell + 0.2)\" height=\"$(cell + 0.2)\" fill=\"$(probability_color(probabilities[index]))\"/>\n")
        end
        for (point, label) in zip(eachrow(data.x_train), data.y_train)
            x = coordinate(point[1])
            y = canvas - margin - (point[2] - low) * plot_size / (high - low)
            fill = label == 1 ? "#f7f7f7" : "#111111"
            print(io, "<circle cx=\"$x\" cy=\"$y\" r=\"3.2\" fill=\"$fill\" stroke=\"#222\" stroke-width=\"0.8\"/>\n")
        end
        print(io, """<rect x="$margin" y="$margin" width="$plot_size" height="$plot_size" fill="none" stroke="#333"/>
<text x="$(canvas / 2)" y="$(canvas - 12)" text-anchor="middle" font-family="sans-serif" font-size="14">Feature 1</text>
<text x="18" y="$(canvas / 2)" text-anchor="middle" transform="rotate(-90 18 $(canvas / 2))" font-family="sans-serif" font-size="14">Feature 2</text>
</svg>
""")
    end
end

train_accuracy = [classification_accuracy(trace, result.data.x_train, result.data.y_train) for trace in result.traces]
test_accuracy = [classification_accuracy(trace, result.data.x_test, result.data.y_test) for trace in result.traces]
across_rate = sum(result.across_acceptance) / length(result.across_acceptance)
within_rate = sum(result.within_acceptance) / length(result.within_acceptance)
posterior_train_accuracy = posterior_accuracy(
    posterior_traces,
    result.data.x_train,
    result.data.y_train,
)
posterior_test_accuracy = posterior_accuracy(
    posterior_traces,
    result.data.x_test,
    result.data.y_test,
)

open(joinpath(output_dir, "trace.csv"), "w") do io
    println(io, "iteration,log_score,width,across_accepted,within_accepted,train_accuracy,test_accuracy")
    for index in eachindex(result.traces)
        println(io, join((
            index,
            result.scores[index],
            result.widths[index],
            result.across_acceptance[index],
            result.within_acceptance[index],
            train_accuracy[index],
            test_accuracy[index],
        ), ','))
    end
end

summary = """
XOR experiment summary
======================
Input: $input_path
Iterations: $(length(result.traces))
Burn-in excluded from posterior average: $burn_in
Final log posterior: $(last(result.scores))
Final hidden width: $(last(result.widths))
Across-dimension acceptance: $(round(100across_rate; digits=2))%
Within-dimension acceptance: $(round(100within_rate; digits=2))%
Mean per-trace training accuracy: $(round(100 * sum(train_accuracy) / length(train_accuracy); digits=2))%
Mean per-trace test accuracy: $(round(100 * sum(test_accuracy) / length(test_accuracy); digits=2))%
Posterior-averaged training accuracy: $(round(100posterior_train_accuracy; digits=2))%
Posterior-averaged test accuracy: $(round(100posterior_test_accuracy; digits=2))%
Width range observed: $(minimum(result.widths))-$(maximum(result.widths))
"""

write(joinpath(output_dir, "summary.txt"), summary)
write_line_plot(joinpath(output_dir, "log_posterior.svg"), result.scores; title="XOR log posterior", y_label="Log posterior")
write_line_plot(joinpath(output_dir, "test_accuracy.svg"), test_accuracy; title="XOR classification accuracy", y_label="Accuracy", lower=0.0, upper=1.0)
write_width_histogram(joinpath(output_dir, "width_histogram.svg"), result.widths, result.settings.maximum_width)
write_decision_surface(joinpath(output_dir, "decision_surface.svg"), posterior_traces, result.data)

print(summary)
println("Analysis written to: $output_dir")
