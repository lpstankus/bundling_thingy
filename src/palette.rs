use clap::ValueEnum;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ColorBy {
    /// Sample the normalized density map.
    #[default]
    Density,
    /// Map input edge IDs across the palette, with one color per edge.
    EdgeId,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum Palette {
    /// Blue to yellow, optimized for red-green color vision deficiencies.
    Cividis,
    /// Purple through teal to yellow.
    #[default]
    Viridis,
    /// Purple through orange to yellow.
    Plasma,
    /// Blue through green and gold to pale pink.
    Batlow,
}

impl Palette {
    pub const fn endpoint_color(self) -> [f32; 4] {
        match self {
            Self::Cividis => [1.0, 0.35, 0.75, 1.0],
            Self::Viridis => [1.0, 0.4, 0.7, 1.0],
            Self::Plasma => [0.15, 0.9, 1.0, 1.0],
            Self::Batlow => [0.1, 0.85, 1.0, 1.0],
        }
    }

    // Sixteen evenly spaced samples per palette, interpolated by the shader.
    // Viridis and plasma by Smith and van der Walt, https://bids.github.io/colormap/
    // Cividis by Nuñez, Anderton, and Renslow, https://doi.org/10.1371/journal.pone.0199239
    // https://github.com/matplotlib/matplotlib/blob/v3.10.7/lib/matplotlib/_cm_listed.py
    // vec4 rows match the 16-byte array stride required by WGSL uniform buffers.
    pub const fn colors(self) -> [[f32; 4]; 16] {
        match self {
            Self::Cividis => [
                [0.000_000, 0.135_112, 0.304_751, 1.0],
                [0.000_000, 0.181_610, 0.421_859, 1.0],
                [0.117_612, 0.225_935, 0.434_308, 1.0],
                [0.208_926, 0.272_546, 0.424_809, 1.0],
                [0.279_411, 0.318_677, 0.423_031, 1.0],
                [0.342_246, 0.364_939, 0.428_559, 1.0],
                [0.401_418, 0.411_790, 0.440_708, 1.0],
                [0.458_366, 0.459_552, 0.460_457, 1.0],
                [0.517_920, 0.508_454, 0.472_707, 1.0],
                [0.582_087, 0.558_670, 0.468_118, 1.0],
                [0.648_222, 0.610_553, 0.454_801, 1.0],
                [0.716_177, 0.664_384, 0.432_386, 1.0],
                [0.785_965, 0.720_438, 0.399_613, 1.0],
                [0.857_809, 0.778_969, 0.353_259, 1.0],
                [0.932_180, 0.840_159, 0.285_880, 1.0],
                [0.995_737, 0.909_344, 0.217_772, 1.0],
            ],
            Self::Viridis => [
                [0.267_004, 0.004_874, 0.329_415, 1.0],
                [0.282_656, 0.100_196, 0.422_160, 1.0],
                [0.277_134, 0.185_228, 0.489_898, 1.0],
                [0.253_935, 0.265_254, 0.529_983, 1.0],
                [0.221_989, 0.339_161, 0.548_752, 1.0],
                [0.190_631, 0.407_061, 0.556_089, 1.0],
                [0.163_625, 0.471_133, 0.558_148, 1.0],
                [0.139_147, 0.533_812, 0.555_298, 1.0],
                [0.120_565, 0.596_422, 0.543_611, 1.0],
                [0.134_692, 0.658_636, 0.517_649, 1.0],
                [0.208_030, 0.718_701, 0.472_873, 1.0],
                [0.327_796, 0.773_980, 0.406_640, 1.0],
                [0.477_504, 0.821_444, 0.318_195, 1.0],
                [0.647_257, 0.858_400, 0.209_861, 1.0],
                [0.824_940, 0.884_720, 0.106_217, 1.0],
                [0.993_248, 0.906_157, 0.143_936, 1.0],
            ],
            Self::Plasma => [
                [0.050_383, 0.029_803, 0.527_975, 1.0],
                [0.200_445, 0.017_902, 0.593_364, 1.0],
                [0.312_543, 0.008_239, 0.635_700, 1.0],
                [0.417_642, 0.000_564, 0.658_390, 1.0],
                [0.517_933, 0.021_563, 0.654_109, 1.0],
                [0.610_667, 0.090_204, 0.619_951, 1.0],
                [0.692_840, 0.165_141, 0.564_522, 1.0],
                [0.764_193, 0.240_396, 0.502_126, 1.0],
                [0.826_588, 0.315_714, 0.441_316, 1.0],
                [0.881_443, 0.392_529, 0.383_229, 1.0],
                [0.928_329, 0.472_975, 0.326_067, 1.0],
                [0.965_024, 0.559_118, 0.268_513, 1.0],
                [0.988_260, 0.652_325, 0.211_364, 1.0],
                [0.994_141, 0.753_137, 0.161_404, 1.0],
                [0.977_995, 0.861_432, 0.142_808, 1.0],
                [0.940_015, 0.975_158, 0.131_326, 1.0],
            ],
            Self::Batlow => BATLOW,
        }
    }
}

// Batlow by Fabio Crameri, Scientific colour maps 8.0, https://www.fabiocrameri.ch/batlow/
// Sampled at sixteen evenly spaced positions from cmcrameri 1.9:
// https://github.com/callumrollo/cmcrameri/blob/v1.9/cmcrameri/cmaps/batlow.txt
/*
MIT License

Copyright (c) 2020 Fabio Crameri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
const BATLOW: [[f32; 4]; 16] = [
    [0.005_193, 0.098_238, 0.349_842, 1.0],
    [0.050_795, 0.196_274, 0.366_684, 1.0],
    [0.069_002, 0.270_922, 0.378_774, 1.0],
    [0.094_462, 0.332_712, 0.385_116, 1.0],
    [0.149_706, 0.386_975, 0.374_449, 1.0],
    [0.237_075, 0.428_325, 0.335_634, 1.0],
    [0.339_469, 0.461_685, 0.279_468, 1.0],
    [0.449_052, 0.493_684, 0.221_074, 1.0],
    [0.570_016, 0.526_186, 0.175_273, 1.0],
    [0.701_421, 0.555_294, 0.185_478, 1.0],
    [0.825_472, 0.577_725, 0.264_197, 1.0],
    [0.928_323, 0.605_212, 0.385_404, 1.0],
    [0.983_574, 0.649_590, 0.538_819, 1.0],
    [0.993_074, 0.699_161, 0.687_519, 1.0],
    [0.989_411, 0.747_997, 0.829_372, 1.0],
    [0.981_354, 0.800_406, 0.981_267, 1.0],
];
