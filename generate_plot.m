clear;

distances = ["66.31"];
result_folders = ["results", "results_avg", "results_1"];
labels = ["two LEDs, single call", "two LEDs, two calls", "one LED"];
sigmas = 0:20;
for i = 1:length(result_folders)
    for j = 1:length(distances)
        for k = sigmas
            name = sprintf("../eye_tracking/%s/%s_%d.txt", result_folders(i), distances(j), k);
            lines = readlines(name);
            for m = 1 : length(lines)-1
                data(m,:) = sscanf(lines(m), '{%f, %f, %f}');
            end
            means(k + 1) = mean(sqrt(data(:, 1) .^ 2 + data(:, 2) .^ 2));
        end
        plot(sigmas, means);
        hold on;
    end
end
hold off;
legend(labels, 'Location', 'southeast');
title('Averaging LED 1 and LED 2');

f = gcf;
exportgraphics(f, 'plot.pdf', 'ContentType', 'vector');