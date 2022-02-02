function cmap = capsuleColorMap()
% Define the colormap used by CamVid dataset.

cmap = [
    182 158 21   % Infllamatory lesion
    81 42 151       % Vascular lesion
    77 170 151   % Healthy tissue
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end