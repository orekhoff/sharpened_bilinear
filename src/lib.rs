use image::{DynamicImage, RgbImage, RgbaImage};

fn srgb_to_lrgb(v: f32) -> f32 {
    if !(v > 0.0) {
        0.0
    } else if v <= 0.04045 {
        v / 12.92
    } else if v < 1.0 {
        ((v + 0.055) / 1.055).powf(2.4)
    } else {
        1.0
    }
}

fn lrgb_to_srgb8(v: f32) -> u8 {
    let v = if !(v > 0.0) {
        0.0
    } else if v <= 0.0031308 {
        12.92 * v
    } else if v < 1.0 {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    } else {
        1.0
    };
    (v * 255.0 + 0.5) as u8
}

/// Processing-friendly image structure
/// with separate channels in ARGB or RGB order
/// in linear color space with alpha premultiplied.
/// 
/// If you are not using the [image] library,
/// you will have to implement the conversion
/// to this structure and back to your image
/// format yourself.
#[derive(Clone, Debug)]
pub struct FullImage {
    pub width: usize,
    pub height: usize,
    pub has_alpha: bool,
    pub channels: Vec<ImageChannel>,
}

impl FullImage {
    pub fn new(width: usize, height: usize, channel_count: usize, has_alpha: bool) -> Self {
        Self {
            width,
            height,
            has_alpha,
            channels: vec![ImageChannel::new(width, height); channel_count],
        }
    }
}

impl From<&DynamicImage> for FullImage {
    fn from(input: &DynamicImage) -> Self {
        let (width, height) = (input.width() as usize, input.height() as usize);
        let has_alpha = input.color().has_alpha();
        let channel_count = if has_alpha { 4 } else { 3 };
        let mut output = FullImage::new(width, height, channel_count, has_alpha);
        if has_alpha {
            let data = input.to_rgba32f().into_vec();
            for i in 0..width * height {
                let r = data[i * 4 + 0];
                let g = data[i * 4 + 1];
                let b = data[i * 4 + 2];
                let a = data[i * 4 + 3];
                output.channels[0].data[i] = a;
                output.channels[1].data[i] = srgb_to_lrgb(r) * a;
                output.channels[2].data[i] = srgb_to_lrgb(g) * a;
                output.channels[3].data[i] = srgb_to_lrgb(b) * a;
            }
        } else {
            let data = input.to_rgb32f().into_vec();
            for i in 0..width * height {
                let r = data[i * 3 + 0];
                let g = data[i * 3 + 1];
                let b = data[i * 3 + 2];
                output.channels[0].data[i] = srgb_to_lrgb(r);
                output.channels[1].data[i] = srgb_to_lrgb(g);
                output.channels[2].data[i] = srgb_to_lrgb(b);
            }
        }
        output
    }
}

impl From<&FullImage> for DynamicImage {
    fn from(input: &FullImage) -> Self {
        if input.channels.len() == 3 && !input.has_alpha {
            let mut buf = vec![0; input.width * input.height * 3];
            for i in 0..input.width * input.height {
                buf[i * 3 + 0] = lrgb_to_srgb8(input.channels[0].data[i]);
                buf[i * 3 + 1] = lrgb_to_srgb8(input.channels[1].data[i]);
                buf[i * 3 + 2] = lrgb_to_srgb8(input.channels[2].data[i]);
            }
            DynamicImage::ImageRgb8(
                RgbImage::from_raw(input.width as u32, input.height as u32, buf).unwrap(),
            )
        } else if input.channels.len() == 4 && input.has_alpha {
            let mut buf = vec![0; input.width * input.height * 4];
            for i in 0..input.width * input.height {
                let a = input.channels[0].data[i] + f32::EPSILON;
                let r = input.channels[1].data[i];
                let g = input.channels[2].data[i];
                let b = input.channels[3].data[i];
                buf[i * 4 + 0] = lrgb_to_srgb8(r / a);
                buf[i * 4 + 1] = lrgb_to_srgb8(g / a);
                buf[i * 4 + 2] = lrgb_to_srgb8(b / a);
                buf[i * 4 + 3] = (a * 255.0 + 0.5) as u8;
            }
            DynamicImage::ImageRgba8(
                RgbaImage::from_raw(input.width as u32, input.height as u32, buf).unwrap(),
            )
        } else {
            panic!("This is not ARGB or RGB image");
        }
    }
}

impl From<FullImage> for DynamicImage {
    fn from(input: FullImage) -> Self {
        (&input).into()
    }
}

/// Separate color channel, dimensions must match the entire image
#[derive(Clone, Debug)]
pub struct ImageChannel {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

#[derive(Copy, Clone, Default)]
struct Area([f32; 9]);

impl Area {
    fn map<F: FnMut(f32) -> f32>(&self, f: &mut F) -> Self {
        let mut out = Self::default();
        for i in 0..9 {
            out.0[i] = f(self.0[i]);
        }
        out
    }
    fn zip_map<F: FnMut(f32, f32) -> f32>(&self, other: &Self, f: &mut F) -> Self {
        let mut out = Self::default();
        for i in 0..9 {
            out.0[i] = f(self.0[i], other.0[i]);
        }
        out
    }
    fn center(&self) -> f32 {
        self.0[4]
    }
    fn borders(&self) -> f32 {
        self.0[1] + self.0[3] + self.0[5] + self.0[7]
    }
    fn corners(&self) -> f32 {
        self.0[0] + self.0[2] + self.0[6] + self.0[8]
    }
    fn integral(&self) -> f32 {
        (self.center() * 36.0 + self.borders() * 6.0 + self.corners()) / 64.0
    }
    fn get(&self, x: usize, y: usize) -> f32 {
        self.0[y * 3 + x]
    }
}

impl ImageChannel {
    pub fn new(width: usize, height: usize) -> Self {
        ImageChannel {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
    fn get_area(&self, x: usize, y: usize) -> Area {
        let x = [x.saturating_sub(1), x, (x + 1).min(self.width - 1)];
        let y = [y.saturating_sub(1), y, (y + 1).min(self.height - 1)];

        Area([
            self.get(x[0], y[0]),
            self.get(x[1], y[0]),
            self.get(x[2], y[0]),
            self.get(x[0], y[1]),
            self.get(x[1], y[1]),
            self.get(x[2], y[1]),
            self.get(x[0], y[2]),
            self.get(x[1], y[2]),
            self.get(x[2], y[2]),
        ])
    }
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        self.data[y * self.width + x] = value
    }
}

fn sharp(
    input_channel: &ImageChannel,
    target: &ImageChannel,
    alpha_channel: Option<&ImageChannel>,
    output_channel: &mut ImageChannel,
) {
    for y in 0..input_channel.height {
        for x in 0..input_channel.width {
            let area = input_channel.get_area(x, y);

            let target = target.get(x, y);
            let borders = area.borders();
            let corners = area.corners();
            let result = (target * 64.0 - borders * 6.0 - corners) / 36.0;

            let max = if let Some(alpha_channel) = alpha_channel {
                alpha_channel.get(x, y)
            } else {
                1.0
            };
            let result = result.clamp(0.0, max);

            output_channel.set(x, y, result);
        }
    }
}

fn adjust_3x3(target: f32, area: Area, alpha: Area) -> Area {
    let current = area.integral();
    if current > target {
        let k = target / current;
        return area.map(&mut |v| v * k);
    }

    let max = alpha.integral();
    if max > current {
        let k = (target - current) / (max - current);
        return area.zip_map(&alpha, &mut |v, a| v * (1.0 - k) + a * k);
    }

    area
}

#[derive(Clone, Debug)]
struct Segment {
    output_index: usize,
    interpolation_factor: f32,
    size: f32,
}

#[derive(Clone, Debug)]
struct IntersectedPixels([Vec<Segment>; 2]);

impl IntersectedPixels {
    fn new(old: usize, new: usize, inp_idx: usize) -> Self {
        let old_div_new = old as f32 / new as f32;
        let new_div_old = 1.0 / old_div_new;

        let before_center = {
            // (idx * new / old).floor()
            let start = (inp_idx * new) / old;
            // ((idx + 0.5) * new / old).ceil()
            // ((idx + 1) * new / (old * 2)).ceil()
            let end = ((inp_idx * 2 + 1) * new).div_ceil(old * 2);

            (start..end)
                .map(|out_idx| {
                    let segment_start = out_idx as f32 * old_div_new;
                    let segment_end = segment_start + old_div_new;

                    let segment_start = segment_start.max(inp_idx as f32);
                    let segment_end = segment_end.min(inp_idx as f32 + 0.5);

                    let size = (segment_end - segment_start) * new_div_old;

                    let center = (segment_start + segment_end) * 0.5;
                    let interpolation_factor = center + 0.5 - inp_idx as f32;

                    Segment {
                        output_index: out_idx,
                        interpolation_factor,
                        size,
                    }
                })
                .collect()
        };

        let after_center = {
            // ((idx + 0.5) * new / old).floor()
            // ((idx + 1) * new / (old * 2)).floor()
            let start = ((inp_idx * 2 + 1) * new) / (old * 2);
            // ((idx + 1) * new / old).ceil()
            let end = ((inp_idx + 1) * new).div_ceil(old);

            (start..end)
                .map(|out_idx| {
                    let segment_start = out_idx as f32 * old_div_new; // 0
                    let segment_end = segment_start + old_div_new; // 2

                    let segment_start = segment_start.max(inp_idx as f32 + 0.5);
                    let segment_end = segment_end.min(inp_idx as f32 + 1.0);

                    let size = (segment_end - segment_start) * new_div_old;

                    let center = (segment_start + segment_end) * 0.5;
                    let lerp_k = center - 0.5 - inp_idx as f32;

                    Segment {
                        output_index: out_idx,
                        interpolation_factor: lerp_k,
                        size,
                    }
                })
                .collect()
        };

        IntersectedPixels([before_center, after_center])
    }
}

fn bilinear_interpolation(a: f32, b: f32, c: f32, d: f32, tx: f32, ty: f32) -> f32 {
    let ab = (b - a) * tx + a;
    let cd = (d - c) * tx + c;
    (cd - ab) * ty + ab
}

/// Performs image resampling.
/// 
/// `input` - something implementing the [Into]<[FullImage]> trait,
/// this trait is implemented for [image::DynamicImage];
/// 
/// `width` and `height` are the dimensions of the output image, must not be zero;
/// 
/// the output type is [FullImage], so the `into()` method must be called to convert it to [image::DynamicImage].
///
/// Usage:
/// ```no_run
/// # let (width, height) = (1, 1);
/// let input_image = image::open("input.png").unwrap();
/// let resized_image: image::DynamicImage =
///     sharpened_bilinear::resize(&input_image, width, height).into();
/// resized_image.save("output.png").unwrap();
/// ```
pub fn resize(input: impl Into<FullImage>, width: usize, height: usize) -> FullImage {
    assert!(width > 0);
    assert!(height > 0);
    
    let input = input.into();
    let mut sharpened_image = input.clone();
    let mut temp_channel = sharpened_image.channels[0].clone();

    let steps = 8;
    // independed channels
    for z in 0..input.channels.len() {
        if input.has_alpha && z != 0 {
            continue;
        }
        for _ in 0..steps {
            let target = &input.channels[z];
            let current_channel = &mut sharpened_image.channels[z];
            sharp(current_channel, target, None, &mut temp_channel);
            sharp(&temp_channel, target, None, current_channel);
        }
    }
    // alpha depended channels
    if input.has_alpha {
        for z in 1..input.channels.len() {
            for _ in 0..steps {
                let target = &input.channels[z];
                let current_channel = &mut sharpened_image.channels[z];
                let max = Some(&input.channels[0]);
                sharp(current_channel, target, max, &mut temp_channel);
                sharp(&temp_channel, target, max, current_channel);
            }
        }
    }

    let intersected_by_x: Vec<_> = (0..input.width)
        .map(|inp_idx| IntersectedPixels::new(input.width, width, inp_idx))
        .collect();

    let intersected_by_y: Vec<_> = (0..input.height)
        .map(|inp_idx| IntersectedPixels::new(input.height, height, inp_idx))
        .collect();

    let mut output_image = FullImage::new(width, height, input.channels.len(), input.has_alpha);

    for y in 0..input.height {
        for x in 0..input.width {
            let mut alpha_area = Area([1.0; 9]);

            let intersected_x = &intersected_by_x[x];
            let intersected_y = &intersected_by_y[y];

            for z in 0..input.channels.len() {
                let area = sharpened_image.channels[z].get_area(x, y);
                let target = input.channels[z].get(x, y);
                let area = adjust_3x3(target, area, alpha_area);
                if input.has_alpha && z == 0 {
                    alpha_area = area;
                }
                for h in 0..2 {
                    for w in 0..2 {
                        for y_segment in intersected_y.0[h].iter() {
                            for x_segment in intersected_x.0[w].iter() {
                                let result = bilinear_interpolation(
                                    area.get(0 + w, 0 + h),
                                    area.get(1 + w, 0 + h),
                                    area.get(0 + w, 1 + h),
                                    area.get(1 + w, 1 + h),
                                    x_segment.interpolation_factor,
                                    y_segment.interpolation_factor,
                                ) * y_segment.size
                                    * x_segment.size;
                                let x = x_segment.output_index;
                                let y = y_segment.output_index;
                                let old = output_image.channels[z].get(x, y);
                                output_image.channels[z].set(x, y, old + result);
                            }
                        }
                    }
                }
            }
        }
    }

    output_image
}
