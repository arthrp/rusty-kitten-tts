//! KittenTTS — local ONNX + voices CLI.

mod espeak_native;
mod model;
mod npz;
mod phonemize;
mod preprocess;

use anyhow::Result;
use clap::Parser;
use model::KittenTts;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "kittentts-rs")]
#[command(about = "KittenTTS text-to-speech (ONNX + voices.npz)")]
struct Args {
    /// Text to synthesize (or use --file)
    #[arg(short, long)]
    text: Option<String>,

    /// Read input text from file (UTF-8)
    #[arg(short = 'F', long)]
    file: Option<PathBuf>,

    #[arg(short, long, default_value = "Bruno")]
    voice: String,

    #[arg(short, long, default_value_t = 1.0)]
    speed: f32,

    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Path to kitten_tts_*.onnx
    #[arg(long, default_value = "../kitten_tts_micro_v0_8.onnx")]
    model: PathBuf,

    /// Path to voices.npz
    #[arg(long, default_value = "voices.npz")]
    voices: PathBuf,

    /// Skip text preprocessing (numbers, currency, etc.)
    #[arg(long)]
    raw_text: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let input_text = match (&args.text, &args.file) {
        (Some(t), None) => t.clone(),
        (None, Some(p)) => std::fs::read_to_string(p)?,
        (Some(_), Some(_)) => {
            anyhow::bail!("use either --text or --file, not both");
        }
        (None, None) => {
            anyhow::bail!("provide --text or --file");
        }
    };

    let mut tts = KittenTts::new(&args.model, &args.voices)?;
    tts.generate_to_file(
        &input_text,
        &args.output,
        &args.voice,
        args.speed,
        !args.raw_text,
    )?;
    eprintln!("Wrote {}", args.output.display());
    Ok(())
}
