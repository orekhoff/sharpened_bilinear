fn main() {
    let args = parse_args();
    let input_image = image::open(args.input).unwrap();

    let (input_width, input_height) = (input_image.width() as f32, input_image.height() as f32);
    let (width, height) = if let Some((width, height)) = args.width.zip(args.height) {
        (width, height)
    } else if let Some(width) = args.width {
        let factor = args.factor.unwrap_or(width / input_width);
        let height = input_height * factor;
        (width, height)
    } else if let Some(height) = args.height {
        let factor = args.factor.unwrap_or(height / input_height);
        let width = input_width * factor;
        (width, height)
    } else {
        let factor = args.factor.unwrap_or(1.0);
        let width = input_width * factor;
        let height = input_height * factor;
        (width, height)
    };
    let width = (width + 0.5).max(1.0) as usize;
    let height = (height + 0.5).max(1.0) as usize;

    let resized: image::DynamicImage =
        sharpened_bilinear::resize(&input_image, width, height).into();
    resized.save(args.output).unwrap();
}

#[derive(Debug)]
struct Args {
    input: String,
    output: String,
    width: Option<f32>,
    height: Option<f32>,
    factor: Option<f32>,
}

const HELP: &str = "\
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
";

fn parse_args() -> Args {
    use lexopt::Arg::{Long, Short, Value};

    let mut input = None;
    let mut output = None;
    let mut width = None;
    let mut height = None;
    let mut factor = None;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next().unwrap() {
        match arg {
            Value(value) => {
                input = Some(
                    value
                        .into_string()
                        .expect("Input path should be correct utf8"),
                );
            }
            Short('o') | Long("output") => {
                output = Some(
                    parser
                        .value()
                        .expect("Output path should be given")
                        .into_string()
                        .expect("Output path should be correct utf8"),
                );
            }
            Short('w') | Long("width") => {
                width = Some(
                    parser
                        .value()
                        .expect("Width value should be given")
                        .to_string_lossy()
                        .parse()
                        .expect("Width value should be number"),
                );
            }
            Short('h') | Long("height") => {
                height = Some(
                    parser
                        .value()
                        .expect("Height value should be given")
                        .to_string_lossy()
                        .parse()
                        .expect("Height value should be number"),
                );
            }
            Short('f') | Long("factor") => {
                factor = Some(
                    parser
                        .value()
                        .expect("Scale factor value should be given")
                        .to_string_lossy()
                        .parse()
                        .expect("Scale factor value should be number"),
                );
            }

            _ => {
                println!("{HELP}");
                panic!("Unexpected argument given {}", arg.unexpected());
            }
        }
    }

    let input = input.unwrap_or_else(|| {
        println!("{HELP}");
        panic!("Input path should be given");
    });

    if width.is_some_and(|n| !(n >= 0.0))
        || height.is_some_and(|n| !(n >= 0.0))
        || factor.is_some_and(|n| !(n >= 0.0))
    {
        println!("{HELP}");
        panic!("Numbers should be positive");
    }

    let output = output.unwrap_or_else(|| format!("{input}_resized.png"));
    Args {
        input,
        output,
        width,
        height,
        factor,
    }
}
