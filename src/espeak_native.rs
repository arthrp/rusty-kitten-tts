//! Minimal FFI to the system `libespeak-ng` (native C library).
//!
//! Requires `libespeak-ng` at link time, e.g. macOS: `brew install espeak-ng`

use anyhow::{anyhow, Result};
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Mutex;

// From speak_lib.h
const ESPEAK_CHARS_UTF8: i32 = 1;
const ESPEAK_PHONEMES_IPA: i32 = 0x02;

// espeak_ng_OUTPUT_MODE
const ENOUTPUT_MODE_SYNCHRONOUS: u32 = 0x0001;

// espeak_ng_STATUS
const ENS_OK: u32 = 0;

#[link(name = "espeak-ng")]
unsafe extern "C" {
    fn espeak_ng_InitializePath(path: *const c_char);
    fn espeak_ng_Initialize(context: *mut c_void) -> u32;
    fn espeak_ng_InitializeOutput(mode: u32, buffer_length: i32, device: *const c_char) -> u32;
    fn espeak_ng_SetVoiceByName(name: *const c_char) -> u32;
    fn espeak_TextToPhonemes(
        textptr: *mut *const c_void,
        textmode: i32,
        phonememode: i32,
    ) -> *const c_char;
    fn espeak_ng_Terminate();
}

static ESPEAK_INIT: Mutex<bool> = Mutex::new(false);

/// Initialize eSpeak NG once (null path → default data directory / `ESPEAK_DATA_PATH`).
pub fn init() -> Result<()> {
    let mut guard = ESPEAK_INIT
        .lock()
        .map_err(|_| anyhow!("espeak init lock poisoned"))?;
    if *guard {
        return Ok(());
    }
    unsafe {
        espeak_ng_InitializePath(std::ptr::null());
        let st = espeak_ng_Initialize(std::ptr::null_mut());
        if st != ENS_OK {
            return Err(anyhow!(
                "espeak_ng_Initialize failed with status {st:#x} (install espeak-ng / set ESPEAK_DATA_PATH)"
            ));
        }
        let st = espeak_ng_InitializeOutput(ENOUTPUT_MODE_SYNCHRONOUS, 0, std::ptr::null());
        if st != ENS_OK {
            return Err(anyhow!("espeak_ng_InitializeOutput failed with status {st:#x}"));
        }
    }
    *guard = true;
    Ok(())
}

/// Optional: call on shutdown (not required for CLI one-shot use).
#[allow(dead_code)]
pub fn terminate() {
    if let Ok(mut g) = ESPEAK_INIT.lock() {
        if *g {
            unsafe { espeak_ng_Terminate() };
            *g = false;
        }
    }
}

/// Set voice by eSpeak name (e.g. `en-us`, `en`).
pub fn set_voice(name: &str) -> Result<()> {
    let c = CString::new(name)?;
    let st = unsafe { espeak_ng_SetVoiceByName(c.as_ptr()) };
    if st != ENS_OK {
        return Err(anyhow!(
            "espeak_ng_SetVoiceByName({name:?}) failed with status {st:#x}"
        ));
    }
    Ok(())
}

/// Raw espeak text → IPA (no punctuation preservation).
fn text_to_ipa_raw(text: &str) -> Result<String> {
    let mut buf = Vec::with_capacity(text.len() + 1);
    buf.extend_from_slice(text.as_bytes());
    buf.push(0);

    let mut ptr: *const c_void = buf.as_ptr().cast();
    let mut out = String::new();
    let mut guard = 0;
    const MAX_CHUNKS: usize = 4096;

    loop {
        if guard >= MAX_CHUNKS {
            return Err(anyhow!("espeak_TextToPhonemes: exceeded {MAX_CHUNKS} chunks (infinite loop?)"));
        }
        guard += 1;

        let pho = unsafe {
            espeak_TextToPhonemes(
                &mut ptr as *mut *const c_void,
                ESPEAK_CHARS_UTF8,
                ESPEAK_PHONEMES_IPA,
            )
        };
        if pho.is_null() {
            break;
        }
        let chunk = unsafe { CStr::from_ptr(pho) }
            .to_string_lossy()
            .trim()
            .to_string();
        if !chunk.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&chunk);
        }

        if ptr.is_null() {
            break;
        }
        let tail = unsafe { CStr::from_ptr(ptr.cast::<c_char>()) };
        if tail.to_bytes().is_empty() {
            break;
        }
    }

    Ok(out)
}

fn is_punct(c: char) -> bool {
    matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '\u{2014}' | '\u{2026}'
        | '\u{00a1}' | '\u{00bf}' | '"' | '\u{00ab}' | '\u{00bb}'
        | '\u{201c}' | '\u{201d}')
}

/// Full UTF-8 text → IPA (UTF-8) with punctuation preserved (like Python
/// `phonemizer` with `preserve_punctuation=True`).
pub fn text_to_ipa(text: &str) -> Result<String> {
    init()?;
    set_voice("en-us")?;

    let trimmed = text.trim();
    let leading: String = trimmed.chars().take_while(|c| is_punct(*c)).collect();
    let trailing: String = trimmed.chars().rev().take_while(|c| is_punct(*c)).collect::<Vec<_>>()
        .into_iter().rev().collect();

    let inner_start = leading.len();
    let inner_end = trimmed.len() - trailing.len();
    let inner = if inner_start < inner_end {
        &trimmed[inner_start..inner_end]
    } else {
        ""
    };

    let ipa = if inner.trim().is_empty() {
        String::new()
    } else {
        text_to_ipa_raw(inner)?
    };

    let mut result = String::new();
    result.push_str(&leading);
    result.push_str(&ipa);
    result.push_str(&trailing);
    Ok(result)
}
