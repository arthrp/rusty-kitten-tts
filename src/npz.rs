//! Minimal NPZ (ZIP of NPY) reader for float32 arrays.

use anyhow::{anyhow, Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use std::collections::HashMap;
use std::io::Read;

/// Load all `.npy` arrays from an NPZ file into a map (key without `.npy` suffix).
pub fn load_npz_float32(path: &std::path::Path) -> Result<HashMap<String, Vec<f32>>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)?;

    let mut out = HashMap::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        if !name.ends_with(".npy") {
            continue;
        }
        let key = name.trim_end_matches(".npy").to_string();
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;
        let data = parse_npy_f32(&buf).with_context(|| format!("parse {name}"))?;
        out.insert(key, data);
    }

    if out.is_empty() {
        return Err(anyhow!("no .npy entries found in NPZ"));
    }
    Ok(out)
}

/// Voice matrices from NPZ: (rows, cols) e.g. (400, 256)
pub fn load_voice_matrix(entry: &[f32], rows: usize, cols: usize) -> Result<Vec<Vec<f32>>> {
    let expected = rows * cols;
    if entry.len() != expected {
        return Err(anyhow!(
            "voice array length {} != {}*{}",
            entry.len(),
            rows,
            cols
        ));
    }
    let mut m = Vec::with_capacity(rows);
    for r in 0..rows {
        let start = r * cols;
        m.push(entry[start..start + cols].to_vec());
    }
    Ok(m)
}

fn parse_npy_f32(buf: &[u8]) -> Result<Vec<f32>> {
    if buf.len() < 10 {
        return Err(anyhow!("npy too short"));
    }
    // Magic \x93NUMPY
    if &buf[0..6] != b"\x93NUMPY" {
        return Err(anyhow!("invalid NPY magic"));
    }
    let major = buf[6];
    let minor = buf[7];
    let (header_len, header_start) = match (major, minor) {
        (1, 0) => (LittleEndian::read_u16(&buf[8..10]) as usize, 10),
        _ => {
            let hl = LittleEndian::read_u32(&buf[8..12]) as usize;
            (hl, 12)
        }
    };
    let header_end = header_start + header_len;
    if header_end > buf.len() {
        return Err(anyhow!("invalid NPY header length"));
    }
    let header = std::str::from_utf8(&buf[header_start..header_end])
        .map_err(|e| anyhow!("npy header utf8: {e}"))?;

    let descr = parse_descr(header).ok_or_else(|| anyhow!("missing descr in header: {header}"))?;
    if descr != "<f4" && descr != "|f4" {
        return Err(anyhow!("expected float32 descr, got {descr:?}"));
    }
    let shape = parse_shape(header).ok_or_else(|| anyhow!("missing shape in header"))?;
    let mut nelem = 1usize;
    for &d in &shape {
        nelem = nelem.checked_mul(d).ok_or_else(|| anyhow!("shape overflow"))?;
    }

    let data_start = header_end;
    let data_len = nelem * 4;
    if data_start + data_len > buf.len() {
        return Err(anyhow!("npy data truncated"));
    }
    let mut v = Vec::with_capacity(nelem);
    let slice = &buf[data_start..data_start + data_len];
    for chunk in slice.chunks_exact(4) {
        v.push(LittleEndian::read_f32(chunk));
    }
    Ok(v)
}

fn parse_descr(header: &str) -> Option<&str> {
    // {'descr': '<f4', ...}
    let key = "'descr': '";
    let start = header.find(key)? + key.len();
    let end = header[start..].find('\'')?;
    Some(&header[start..start + end])
}

fn parse_shape(header: &str) -> Option<Vec<usize>> {
    let key = "'shape': (";
    let start = header.find(key)? + key.len();
    let rest = &header[start..];
    let end_paren = rest.find(')')?;
    let inside = rest[..end_paren].trim();
    if inside.is_empty() {
        return Some(vec![]);
    }
    let mut dims = Vec::new();
    for part in inside.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        dims.push(p.parse().ok()?);
    }
    Some(dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_inline_known_header() {
        // minimal valid header for (2,) float32
        let h = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
        let mut buf = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];
        let hl = h.len() as u16;
        buf.extend_from_slice(&hl.to_le_bytes());
        buf.extend_from_slice(h.as_bytes());
        buf.extend_from_slice(&0.25f32.to_le_bytes());
        buf.extend_from_slice(&0.5f32.to_le_bytes());
        let v = parse_npy_f32(&buf).unwrap();
        assert_eq!(v.len(), 2);
        assert!((v[0] - 0.25).abs() < 1e-6);
    }
}
