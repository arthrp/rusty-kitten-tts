//! Phonemization via native `libespeak-ng` + tokenization (see Python pipeline).

use crate::espeak_native;
use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Text → IPA using system eSpeak NG (`espeak_TextToPhonemes`, IPA mode).
pub fn text_to_ipa(text: &str) -> Result<String> {
    espeak_native::text_to_ipa(text)
}

/// Same tokenization as Python `basic_english_tokenize` — Unicode word characters.
fn basic_english_tokenize(text: &str) -> Vec<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?u)\w+|[^\w\s]").expect("tokenize regex"));
    re.find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Character → model index (see Python `TextCleaner`).
pub struct TextCleaner {
    map: HashMap<char, i64>,
}

impl TextCleaner {
    pub fn new() -> Self {
        let pad = "$";
        let punctuation = ";:,.!?¡¿—…\"«»\u{201c}\u{201d} ";
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        let letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

        let mut symbols: Vec<char> = Vec::new();
        symbols.extend(pad.chars());
        symbols.extend(punctuation.chars());
        symbols.extend(letters.chars());
        symbols.extend(letters_ipa.chars());

        let mut map = HashMap::new();
        for (i, c) in symbols.into_iter().enumerate() {
            map.insert(c, i as i64);
        }
        Self { map }
    }

    pub fn encode_chars(&self, text: &str) -> Vec<i64> {
        let mut v = Vec::new();
        for c in text.chars() {
            if let Some(&idx) = self.map.get(&c) {
                v.push(idx);
            }
        }
        v
    }
}

impl Default for TextCleaner {
    fn default() -> Self {
        Self::new()
    }
}

/// Full pipeline: text (already preprocessed) → IPA → space‑joined tokens → ids with BOS/EOS.
pub fn text_to_input_ids(preprocessed_text: &str, cleaner: &TextCleaner) -> Result<Vec<i64>> {
    let ipa = text_to_ipa(preprocessed_text)?;
    let toks = basic_english_tokenize(&ipa);
    let joined = toks.join(" ");
    let mut ids = cleaner.encode_chars(&joined);
    ids.insert(0, 0);
    ids.push(10);
    ids.push(0);
    Ok(ids)
}
