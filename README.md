# Sharpened Bilinear

Sharpened Bilinear is an image resizing library and command-line tool
that uses a modified bilinear interpolation algorithm to produce sharp,
accurate results. It supports resizing with premultiplied alpha and
operates in linear space to avoid brightness distortion.

## Installation

To use Sharpened Bilinear as a command-line tool, install it with Cargo:

```bash
cargo install sharpened_bilinear
```

To use Sharpened Bilinear as a library, add the following to your `Cargo.toml`:

```toml
[dependencies]
sharpened_bilinear = "1.0.0"
```

## Usage

### Command-line tool

```
USAGE: sharpened_bilinear <INPUT> [OPTIONS]

ARGS:
  <INPUT>         Input file path (required)

OPTIONS:
  --output PATH   Output file path
  --factor NUMBER Scale factor
  --width  NUMBER Output image width in pixels
  --height NUMBER Output image height in pixels

Defaults:
  sharpened_bilinear <INPUT> -o <INPUT>_resized.png -f 1.0

If only one of the keys '-w' or '-h' is given, the
the second dimension preserves the aspect ratio or
determined from the '-f' key if factor given.
```

### Library

```rust
let input_image = image::open("input.png").unwrap();
let resized_image: image::DynamicImage =
    sharpened_bilinear::resize(&input_image, width, height).into();
resized_image.save("output.png").unwrap();
```

## License

Sharpened Bilinear is licensed under the MIT License.
