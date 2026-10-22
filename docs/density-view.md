# Density view

Path colors follow the normalized density of nearby paths, using the Cividis colormap. Paths use 10% opacity so overlaps stand out against the dark background.

![French airline graph with brighter dense bundles and fainter isolated paths](images/density-fr-airlines.webp)

This image shows `data/fr-airlines.trl`, rendered at 1600 × 1600 and resized for display. Gaussian blur uses σ = 6 texels with a radius of 18 texels.

Run the windowed viewer with `cargo run`. Export the same graph with:

```sh
cargo run -- --headless --resample --validate-gpu --input data/fr-airlines.trl --out out/fr-airlines
```

The export writes `out/fr-airlines/original.png`.

The density map matches the output dimensions and rebuilds when the window resizes. Only paths with at least one nonzero segment contribute. A compute shader samples each path at intervals of one texel and rounds each sample to its nearest grid cell. All paths increment one shared counter grid with atomic additions. Two Gaussian blur passes smooth the counts, then normalization divides the field by its blurred maximum. The fragment shader samples that texture with bilinear filtering.

These crops show ten overlapping visible paths alongside 1,026 degenerate paths. Before the fix, the invisible paths raised the normalization maximum and darkened the visible paths. Excluding them restores the visible paths' peak density color. Both renders use the same blur and opacity.

![Cropped visible paths before and after excluding degenerate edges from density](images/density-degenerate-edges.webp)

Higher resolutions increase map memory and sampling work. Dense bundles can contend on the shared counters. Colors show density relative to each graph's own maximum, so they do not give an absolute comparison between graphs.
