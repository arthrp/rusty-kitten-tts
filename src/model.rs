//! ONNX inference, voice tables, chunking, WAV export.

use crate::npz::{load_npz_float32, load_voice_matrix};
use crate::phonemize::{text_to_input_ids, TextCleaner};
use crate::preprocess::TextPreprocessor;
use anyhow::{anyhow, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;

const SAMPLE_RATE: u32 = 24_000;
const TRIM_SAMPLES: usize = 5000;
const CHUNK_MAX: usize = 400;

pub const VOICE_IDS: [&str; 8] = [
    "expr-voice-2-m",
    "expr-voice-2-f",
    "expr-voice-3-m",
    "expr-voice-3-f",
    "expr-voice-4-m",
    "expr-voice-4-f",
    "expr-voice-5-m",
    "expr-voice-5-f",
];

pub const VOICE_ALIASES: [(&str, &str); 8] = [
    ("Bella", "expr-voice-2-m"),
    ("Jasper", "expr-voice-2-f"),
    ("Luna", "expr-voice-3-m"),
    ("Bruno", "expr-voice-3-f"),
    ("Rosie", "expr-voice-4-m"),
    ("Hugo", "expr-voice-4-f"),
    ("Kiki", "expr-voice-5-f"),
    ("Leo", "expr-voice-5-m"),
];

fn ensure_punctuation(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return text.to_string();
    }
    let last = text.chars().last().unwrap();
    if matches!(last, '.' | '!' | '?' | ',' | ';' | ':') {
        text.to_string()
    } else {
        format!("{text},")
    }
}

fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let re = regex::Regex::new(r"[.!?]+").expect("sentence split");
    let mut chunks = Vec::new();
    for sentence in re.split(text) {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }
        if sentence.chars().count() <= max_len {
            chunks.push(ensure_punctuation(sentence));
        } else {
            let mut temp = String::new();
            for word in sentence.split_whitespace() {
                let add = if temp.is_empty() {
                    word.len()
                } else {
                    temp.len() + 1 + word.len()
                };
                if add <= max_len {
                    if temp.is_empty() {
                        temp.push_str(word);
                    } else {
                        temp.push(' ');
                        temp.push_str(word);
                    }
                } else {
                    if !temp.is_empty() {
                        chunks.push(ensure_punctuation(temp.trim()));
                    }
                    temp = word.to_string();
                }
            }
            if !temp.is_empty() {
                chunks.push(ensure_punctuation(temp.trim()));
            }
        }
    }
    chunks
}

pub struct KittenTts {
    session: Session,
    voices: HashMap<String, Vec<Vec<f32>>>,
    preprocessor: TextPreprocessor,
    cleaner: TextCleaner,
    speed_priors: HashMap<String, f32>,
    aliases: HashMap<String, String>,
}

impl KittenTts {
    pub fn new(model_path: &Path, voices_path: &Path) -> Result<Self> {
        Self::with_options(model_path, voices_path, HashMap::new(), default_aliases())
    }

    pub fn with_options(
        model_path: &Path,
        voices_path: &Path,
        speed_priors: HashMap<String, f32>,
        extra_aliases: HashMap<String, String>,
    ) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("onnx session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap_or_else(|e| e.recover())
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("load ONNX {}: {e}", model_path.display()))?;

        let flat = load_npz_float32(voices_path)?;
        let mut voices = HashMap::new();
        for (key, data) in flat {
            let matrix = if data.len() == 400 * 256 {
                load_voice_matrix(&data, 400, 256)?
            } else if data.len() % 256 == 0 {
                let rows = data.len() / 256;
                load_voice_matrix(&data, rows, 256)?
            } else {
                return Err(anyhow!(
                    "voice {key}: unexpected length {}",
                    data.len()
                ));
            };
            voices.insert(key, matrix);
        }

        let mut aliases = default_aliases();
        aliases.extend(extra_aliases);

        Ok(Self {
            session,
            voices,
            preprocessor: TextPreprocessor::kitten_onnx_default(),
            cleaner: TextCleaner::new(),
            speed_priors,
            aliases,
        })
    }

    pub fn resolve_voice(&self, voice: &str) -> Result<String> {
        let v = self
            .aliases
            .get(voice)
            .cloned()
            .unwrap_or_else(|| voice.to_string());
        if !VOICE_IDS.iter().any(|&id| id == v) {
            return Err(anyhow!(
                "voice {voice:?} not available; use one of {:?} or {:?}",
                VOICE_IDS,
                VOICE_ALIASES.iter().map(|(a, _)| *a).collect::<Vec<_>>()
            ));
        }
        if !self.voices.contains_key(&v) {
            return Err(anyhow!("voice embedding {v:?} missing from voices.npz"));
        }
        Ok(v)
    }

    fn speed_for(&self, voice: &str, speed: f32) -> f32 {
        self.speed_priors
            .get(voice)
            .map(|p| speed * p)
            .unwrap_or(speed)
    }

    fn infer_chunk(&mut self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let voice_rows = self
            .voices
            .get(voice)
            .ok_or_else(|| anyhow!("unknown voice {voice}"))?;
        let n_rows = voice_rows.len();
        let ref_id = text.len().min(n_rows.saturating_sub(1));
        let style_row = &voice_rows[ref_id];

        let input_ids = text_to_input_ids(text, &self.cleaner)?;

        let n = input_ids.len() as i64;
        let ids_tensor = Tensor::from_array((vec![1, n], input_ids))?;
        let style_tensor = Tensor::from_array((vec![1i64, 256], style_row.clone()))?;
        let speed_tensor = Tensor::from_array((vec![1i64], vec![speed]))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids_tensor,
            "style" => style_tensor,
            "speed" => speed_tensor
        ])?;

        let audio = outputs[0].try_extract_tensor::<f32>()?;
        let (_, samples) = audio;
        let n_out = samples.len().saturating_sub(TRIM_SAMPLES);
        Ok(samples[..n_out].to_vec())
    }

    pub fn generate(&mut self, text: &str, voice: &str, speed: f32, clean_text: bool) -> Result<Vec<f32>> {
        let voice = self.resolve_voice(voice)?;
        let speed = self.speed_for(&voice, speed);

        let text = if clean_text {
            self.preprocessor.process(text)
        } else {
            text.to_string()
        };

        let mut out: Vec<f32> = Vec::new();
        for chunk in chunk_text(&text, CHUNK_MAX) {
            let mut part = self.infer_chunk(&chunk, &voice, speed)?;
            out.append(&mut part);
        }
        Ok(out)
    }

    pub fn generate_to_file(
        &mut self,
        text: &str,
        out_path: &Path,
        voice: &str,
        speed: f32,
        clean_text: bool,
    ) -> Result<()> {
        let samples = self.generate(text, voice, speed, clean_text)?;
        write_wav_f32(out_path, &samples, SAMPLE_RATE)?;
        Ok(())
    }
}

fn default_aliases() -> HashMap<String, String> {
    VOICE_ALIASES
        .iter()
        .map(|(a, v)| (a.to_string(), v.to_string()))
        .collect()
}

fn write_wav_f32(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut w = WavWriter::create(path, spec).with_context(|| path.display().to_string())?;
    for &s in samples {
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}
