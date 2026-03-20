//! Port of Python `kittentts/preprocess.py` — `TextPreprocessor` pipeline.

use fancy_regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

// ─── numbers ─────────────────────────────────────────────

const ONES: &[&str] = &[
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];

const TENS: &[&str] = &[
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

const SCALE: &[&str] = &["", "thousand", "million", "billion", "trillion"];

fn ordinal_exceptions() -> &'static [(&'static str, &'static str)] {
    &[
        ("one", "first"),
        ("two", "second"),
        ("three", "third"),
        ("four", "fourth"),
        ("five", "fifth"),
        ("six", "sixth"),
        ("seven", "seventh"),
        ("eight", "eighth"),
        ("nine", "ninth"),
        ("twelve", "twelfth"),
    ]
}

fn three_digits_to_words(n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    let mut parts = Vec::new();
    let hundreds = n / 100;
    let remainder = n % 100;
    if hundreds > 0 {
        parts.push(format!("{} hundred", ONES[hundreds]));
    }
    if remainder < 20 {
        if remainder > 0 {
            parts.push(ONES[remainder].to_string());
        }
    } else {
        let tens_word = TENS[remainder / 10];
        let ones_word = ONES[remainder % 10];
        if ones_word.is_empty() {
            parts.push(tens_word.to_string());
        } else {
            parts.push(format!("{tens_word}-{ones_word}"));
        }
    }
    parts.join(" ")
}

pub fn number_to_words(n: i64) -> String {
    if n == 0 {
        return "zero".into();
    }
    if n < 0 {
        return format!("negative {}", number_to_words(-n));
    }
    let n = n as u64;

    if (100..=9999).contains(&n) && n % 100 == 0 && n % 1000 != 0 {
        let hundreds = (n / 100) as usize;
        if hundreds < 20 {
            return format!("{} hundred", ONES[hundreds]);
        }
    }

    let mut parts: Vec<String> = Vec::new();
    let mut x = n;
    let mut i = 0;
    while i < SCALE.len() && x > 0 {
        let chunk = (x % 1000) as usize;
        if chunk > 0 {
            let chunk_words = three_digits_to_words(chunk);
            let scale_word = SCALE[i];
            if scale_word.is_empty() {
                parts.push(chunk_words);
            } else {
                parts.push(format!("{chunk_words} {scale_word}"));
            }
        }
        x /= 1000;
        i += 1;
        if x == 0 {
            break;
        }
    }
    parts.into_iter().rev().collect::<Vec<_>>().join(" ").trim().to_string()
}

pub fn float_to_words(value: &str) -> String {
    let text = value.trim();
    let negative = text.starts_with('-');
    let text = if negative { &text[1..] } else { text };

    let result = if let Some(dot) = text.find('.') {
        let int_part = &text[..dot];
        let dec_part = &text[dot + 1..];
        let int_words = if int_part.is_empty() {
            "zero".to_string()
        } else {
            number_to_words(int_part.parse::<i64>().unwrap_or(0))
        };
        let digit_map: Vec<&str> = ["zero"]
            .into_iter()
            .chain(ONES[1..].iter().copied())
            .collect();
        let mut dec_words = Vec::new();
        for ch in dec_part.chars() {
            if let Some(d) = ch.to_digit(10) {
                dec_words.push(digit_map[d as usize]);
            }
        }
        format!("{} point {}", int_words, dec_words.join(" "))
    } else {
        number_to_words(text.parse::<i64>().unwrap_or(0))
    };

    if negative {
        format!("negative {result}")
    } else {
        result
    }
}

fn roman_to_int(s: &str) -> i64 {
    let val = |c: u8| match c {
        b'I' => 1,
        b'V' => 5,
        b'X' => 10,
        b'L' => 50,
        b'C' => 100,
        b'D' => 500,
        b'M' => 1000,
        _ => 0,
    };
    let b = s.as_bytes();
    let mut result = 0i64;
    let mut prev = 0i64;
    for &ch in b.iter().rev() {
        let curr = val(ch.to_ascii_uppercase());
        result += if curr >= prev { curr } else { -curr };
        prev = curr;
    }
    result
}

// ─── compiled regexes ────────────────────────────────────

static RE_URL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://\S+|www\.\S+").unwrap());
static RE_EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b").unwrap()
});
static RE_HASHTAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"#\w+").unwrap());
static RE_MENTION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@\w+").unwrap());
static RE_HTML: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
static RE_PUNCT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[^\w\s.,?!;:\-\u{2014}\u{2013}\u{2026}]").unwrap()
});
static RE_SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());
static RE_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?<![a-zA-Z])-?[\d,]+(?:\.\d+)?").unwrap());
static RE_ORDINAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(\d+)(st|nd|rd|th)\b").unwrap());
static RE_PERCENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(-?[\d,]+(?:\.\d+)?)\s*%").unwrap());
static RE_CURRENCY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"([$€£¥₹₩₿])\s*([\d,]+(?:\.\d+)?)\s*([KMBT])?(?![a-zA-Z\d])").unwrap()
});
static RE_TIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?\b").unwrap()
});
static RE_RANGE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?<!\w)(\d+)-(\d+)(?!\w)").unwrap());
static RE_MODEL_VER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b([a-zA-Z][a-zA-Z0-9]*)-(\d[\d.]*)(?=[^\d.]|$)").unwrap()
});
static RE_UNIT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(km|kg|mg|ml|gb|mb|kb|tb|hz|khz|mhz|ghz|mph|kph|°[cCfF]|[cCfF]°|ms|ns|µs)\b")
        .unwrap()
});
static RE_SCALE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?<![a-zA-Z])(\d+(?:\.\d+)?)\s*([KMBT])(?![a-zA-Z\d])").unwrap()
});
static RE_SCI: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?<![a-zA-Z\d])(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![a-zA-Z\d])").unwrap()
});
static RE_FRACTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(\d+)\s*/\s*(\d+)\b").unwrap());
static RE_DECADE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(\d{1,3})0s\b").unwrap());
static RE_LEAD_DEC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?<!\d)\.([\d])").unwrap());
static RE_ROMAN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(M{0,4})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b").unwrap()
});
static RE_TITLE_WORDS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(war|chapter|part|volume|act|scene|book|section|article|king|queen|pope|louis|henry|edward|george|william|james|phase|round|level|stage|class|type|version|episode|season)\b").unwrap()
});
static RE_IP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b").unwrap()
});
static RE_NEG_LEAD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?<!\d)(-)\.([\d])").unwrap());
static RE_PHONE11: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?<!\d-)(?<!\d)\b(\d{1,2})-(\d{3})-(\d{3})-(\d{4})\b(?!-\d)").unwrap()
});
static RE_PHONE10: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?<!\d-)(?<!\d)\b(\d{3})-(\d{3})-(\d{4})\b(?!-\d)").unwrap()
});
static RE_PHONE7: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?<!\d-)\b(\d{3})-(\d{4})\b(?!-\d)").unwrap());

fn ordinal_suffix(n: i64) -> String {
    let word = number_to_words(n);
    let (prefix, last, joiner) = if let Some((p, l)) = word.rsplit_once('-') {
        (p.to_string(), l.to_string(), "-".to_string())
    } else if let Some((p, l)) = word.rsplit_once(' ') {
        if p.is_empty() {
            (String::new(), l.to_string(), String::new())
        } else {
            (p.to_string(), l.to_string(), " ".to_string())
        }
    } else {
        (String::new(), word, String::new())
    };

    let last_ord = ordinal_exceptions()
        .iter()
        .find(|(b, _)| *b == last.as_str())
        .map(|(_, o)| (*o).to_string())
        .unwrap_or_else(|| {
            if last.ends_with('t') {
                format!("{last}h")
            } else if last.ends_with('e') {
                format!("{}th", &last[..last.len() - 1])
            } else {
                format!("{last}th")
            }
        });

    if prefix.is_empty() {
        last_ord
    } else {
        format!("{prefix}{joiner}{last_ord}")
    }
}

pub fn expand_ordinals(text: &str) -> String {
    RE_ORDINAL
        .replace_all(text, |caps: &fancy_regex::Captures| {
            ordinal_suffix(caps[1].parse::<i64>().unwrap_or(0))
        })
        .into_owned()
}

pub fn expand_percentages(text: &str) -> String {
    RE_PERCENT
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let raw = caps[1].replace(',', "");
            if raw.contains('.') {
                format!("{} percent", float_to_words(&raw))
            } else {
                format!(
                    "{} percent",
                    number_to_words(raw.parse::<i64>().unwrap_or(0))
                )
            }
        })
        .into_owned()
}

fn currency_scale_word(c: &str) -> &'static str {
    match c {
        "K" => "thousand",
        "M" => "million",
        "B" => "billion",
        "T" => "trillion",
        _ => "",
    }
}

pub fn expand_currency(text: &str) -> String {
    let sym_unit = |s: &str| -> &str {
        match s {
            "$" => "dollar",
            "€" => "euro",
            "£" => "pound",
            "¥" => "yen",
            "₹" => "rupee",
            "₩" => "won",
            "₿" => "bitcoin",
            _ => "",
        }
    };

    RE_CURRENCY
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let symbol = &caps[1];
            let raw = caps[2].replace(',', "");
            let scale_suffix = caps.get(3).map(|m| m.as_str());
            let unit = sym_unit(symbol);

            if let Some(suf) = scale_suffix {
                let scale_word = currency_scale_word(suf);
                let num = if raw.contains('.') {
                    float_to_words(&raw)
                } else {
                    number_to_words(raw.parse::<i64>().unwrap_or(0))
                };
                let plural = if unit.is_empty() { "" } else { "s" };
                return format!("{num} {scale_word} {unit}{plural}").trim().to_string();
            }

            if raw.contains('.') {
                let parts: Vec<&str> = raw.splitn(2, '.').collect();
                let int_part = parts[0];
                let dec_part = parts[1];
                let mut dec_two: String = dec_part.chars().take(2).collect();
                while dec_two.len() < 2 {
                    dec_two.push('0');
                }
                let dec_val: i64 = dec_two.parse().unwrap_or(0);
                let int_words = number_to_words(int_part.parse::<i64>().unwrap_or(0));
                let mut result = if unit.is_empty() {
                    int_words
                } else {
                    format!("{int_words} {unit}s")
                };
                if dec_val > 0 {
                    let cents = number_to_words(dec_val);
                    let cent_plural = if dec_val != 1 { "s" } else { "" };
                    result.push_str(&format!(" and {cents} cent{cent_plural}"));
                }
                result
            } else {
                let val = raw.parse::<i64>().unwrap_or(0);
                let words = number_to_words(val);
                if unit.is_empty() {
                    words
                } else {
                    let plural = if val != 1 { "s" } else { "" };
                    format!("{words} {unit}{plural}")
                }
            }
        })
        .into_owned()
}

pub fn expand_time(text: &str) -> String {
    RE_TIME
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let h: i64 = caps[1].parse().unwrap_or(0);
            let mins: i64 = caps[2].parse().unwrap_or(0);
            let ampm = caps.get(4).map(|m| format!(" {}", m.as_str().to_lowercase()));
            let h_words = number_to_words(h);
            if mins == 0 {
                if ampm.is_none() {
                    format!("{h_words} hundred")
                } else {
                    format!("{h_words}{}", ampm.unwrap_or_default())
                }
            } else if mins < 10 {
                format!(
                    "{h_words} oh {}{}",
                    number_to_words(mins),
                    ampm.unwrap_or_default()
                )
            } else {
                format!(
                    "{h_words} {}{}",
                    number_to_words(mins),
                    ampm.unwrap_or_default()
                )
            }
        })
        .into_owned()
}

pub fn expand_ranges(text: &str) -> String {
    RE_RANGE
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let lo = number_to_words(caps[1].parse::<i64>().unwrap_or(0));
            let hi = number_to_words(caps[2].parse::<i64>().unwrap_or(0));
            format!("{lo} to {hi}")
        })
        .into_owned()
}

pub fn expand_model_names(text: &str) -> String {
    RE_MODEL_VER
        .replace_all(text, |caps: &fancy_regex::Captures| {
            format!("{} {}", &caps[1], &caps[2])
        })
        .into_owned()
}

pub fn expand_units(text: &str) -> String {
    let unit_map: &[(&str, &str)] = &[
        ("km", "kilometers"),
        ("kg", "kilograms"),
        ("mg", "milligrams"),
        ("ml", "milliliters"),
        ("gb", "gigabytes"),
        ("mb", "megabytes"),
        ("kb", "kilobytes"),
        ("tb", "terabytes"),
        ("hz", "hertz"),
        ("khz", "kilohertz"),
        ("mhz", "megahertz"),
        ("ghz", "gigahertz"),
        ("mph", "miles per hour"),
        ("kph", "kilometers per hour"),
        ("ms", "milliseconds"),
        ("ns", "nanoseconds"),
        ("µs", "microseconds"),
        ("°c", "degrees Celsius"),
        ("c°", "degrees Celsius"),
        ("°f", "degrees Fahrenheit"),
        ("f°", "degrees Fahrenheit"),
    ];
    RE_UNIT
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let raw = caps[1].to_string();
            let unit = caps[2].to_lowercase();
            let expanded = unit_map
                .iter()
                .find(|(k, _)| *k == unit.as_str())
                .map(|(_, v)| *v)
                .unwrap_or(caps[2].into());
            let num = if raw.contains('.') {
                float_to_words(&raw)
            } else {
                number_to_words(raw.parse::<i64>().unwrap_or(0))
            };
            format!("{num} {expanded}")
        })
        .into_owned()
}

pub fn expand_roman_numerals(text: &str) -> String {
    RE_ROMAN
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let roman = caps[0].to_string();
            let start = caps.get(0).map(|m| m.start()).unwrap_or(0);
            if roman.trim().is_empty() {
                return roman;
            }
            let bytes = roman.as_bytes();
            if bytes.len() == 1 && matches!(bytes[0], b'I' | b'V' | b'X') {
                let preceding = &text[start.saturating_sub(30)..start];
                if !RE_TITLE_WORDS.is_match(preceding).unwrap_or(false) {
                    return roman;
                }
            }
            let val = roman_to_int(&roman);
            if val == 0 {
                roman
            } else {
                number_to_words(val)
            }
        })
        .into_owned()
}

pub fn normalize_leading_decimals(text: &str) -> String {
    let t = RE_NEG_LEAD.replace_all(text, "${1}0.${2}");
    RE_LEAD_DEC
        .replace_all(&t, "0.${1}")
        .into_owned()
}

pub fn expand_scientific_notation(text: &str) -> String {
    RE_SCI
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let coeff_raw = caps[1].to_string();
            let exp: i64 = caps[2].parse().unwrap_or(0);
            let coeff_words = if coeff_raw.contains('.') {
                float_to_words(&coeff_raw)
            } else {
                number_to_words(coeff_raw.parse::<i64>().unwrap_or(0))
            };
            let exp_words = number_to_words(exp.unsigned_abs() as i64);
            let sign = if exp < 0 { "negative " } else { "" };
            format!("{coeff_words} times ten to the {sign}{exp_words}")
        })
        .into_owned()
}

pub fn expand_scale_suffixes(text: &str) -> String {
    let m: &[(&str, &str)] = &[
        ("K", "thousand"),
        ("M", "million"),
        ("B", "billion"),
        ("T", "trillion"),
    ];
    RE_SCALE
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let raw = caps[1].to_string();
            let suffix = caps[2].to_string();
            let scale_word = m
                .iter()
                .find(|(k, _)| *k == suffix.as_str())
                .map(|(_, v)| v.to_string())
                .unwrap_or_else(|| suffix.clone());
            let num = if raw.contains('.') {
                float_to_words(&raw)
            } else {
                number_to_words(raw.parse::<i64>().unwrap_or(0))
            };
            format!("{num} {scale_word}")
        })
        .into_owned()
}

pub fn expand_fractions(text: &str) -> String {
    RE_FRACTION
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let num = caps[1].parse::<i64>().unwrap_or(0);
            let den = caps[2].parse::<i64>().unwrap_or(0);
            if den == 0 {
                return caps[0].to_string();
            }
            let num_words = number_to_words(num);
            let denom_word = if den == 2 {
                if num == 1 {
                    "half".into()
                } else {
                    "halves".into()
                }
            } else if den == 4 {
                if num == 1 {
                    "quarter".into()
                } else {
                    "quarters".into()
                }
            } else {
                let mut s = ordinal_suffix(den);
                if num != 1 {
                    s.push('s');
                }
                s
            };
            format!("{num_words} {denom_word}")
        })
        .into_owned()
}

pub fn expand_decades(text: &str) -> String {
    let decade_map: &[&str] = &[
        "hundreds", "tens", "twenties", "thirties", "forties", "fifties", "sixties", "seventies",
        "eighties", "nineties",
    ];
    RE_DECADE
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let base: i64 = caps[1].parse().unwrap_or(0);
            let decade_digit = (base % 10) as usize;
            let decade_word = decade_map.get(decade_digit).copied().unwrap_or("");
            if base < 10 {
                decade_word.to_string()
            } else {
                let century_part = base / 10;
                format!(
                    "{} {decade_word}",
                    number_to_words(century_part)
                )
            }
        })
        .into_owned()
}

fn digit_word(ch: u8) -> &'static str {
    match ch {
        b'0' => "zero",
        b'1' => "one",
        b'2' => "two",
        b'3' => "three",
        b'4' => "four",
        b'5' => "five",
        b'6' => "six",
        b'7' => "seven",
        b'8' => "eight",
        b'9' => "nine",
        _ => "",
    }
}

pub fn expand_ip_addresses(text: &str) -> String {
    RE_IP
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let o = |s: &str| {
                s.bytes()
                    .map(|b| digit_word(b))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            format!(
                "{} dot {} dot {} dot {}",
                o(&caps[1]),
                o(&caps[2]),
                o(&caps[3]),
                o(&caps[4])
            )
        })
        .into_owned()
}

fn digits_spoken(s: &str) -> String {
    s.bytes()
        .map(|b| digit_word(b))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn expand_phone_numbers(text: &str) -> String {
    let text = RE_PHONE11.replace_all(text, |caps: &fancy_regex::Captures| {
        [
            digits_spoken(&caps[1]),
            digits_spoken(&caps[2]),
            digits_spoken(&caps[3]),
            digits_spoken(&caps[4]),
        ]
        .join(" ")
    });
    let text = RE_PHONE10.replace_all(&text, |caps: &fancy_regex::Captures| {
        [
            digits_spoken(&caps[1]),
            digits_spoken(&caps[2]),
            digits_spoken(&caps[3]),
        ]
        .join(" ")
    });
    RE_PHONE7
        .replace_all(&text, |caps: &fancy_regex::Captures| {
            [digits_spoken(&caps[1]), digits_spoken(&caps[2])].join(" ")
        })
        .into_owned()
}

pub fn replace_numbers(text: &str, replace_floats: bool) -> String {
    RE_NUMBER
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let raw = caps[0].replace(',', "");
            if raw.contains('.') && replace_floats {
                float_to_words(&raw)
            } else if raw.contains('.') && !replace_floats {
                caps[0].to_string()
            } else {
                match raw.parse::<f64>() {
                    Ok(f) => number_to_words(f as i64),
                    Err(_) => caps[0].to_string(),
                }
            }
        })
        .into_owned()
}

pub fn expand_contractions(text: &str) -> String {
    let mut s = text.to_string();
    let pairs: &[(&str, &str)] = &[
        (r"(?i)\bcan't\b", "cannot"),
        (r"(?i)\bwon't\b", "will not"),
        (r"(?i)\bshan't\b", "shall not"),
        (r"(?i)\bain't\b", "is not"),
        (r"(?i)\blet's\b", "let us"),
    ];
    for (pat, rep) in pairs {
        let re = Regex::new(pat).unwrap();
        s = re.replace_all(&s, *rep).into_owned();
    }
    let re = Regex::new(r"(?i)\b(\w+)n't\b").unwrap();
    s = re
        .replace_all(&s, |c: &fancy_regex::Captures| {
            format!("{} not", c.get(1).map(|m| m.as_str()).unwrap_or(""))
        })
        .into_owned();
    let re = Regex::new(r"(?i)\b(\w+)'re\b").unwrap();
    s = re
        .replace_all(&s, |c: &fancy_regex::Captures| {
            format!("{} are", c.get(1).map(|m| m.as_str()).unwrap_or(""))
        })
        .into_owned();
    let re = Regex::new(r"(?i)\b(\w+)'ve\b").unwrap();
    s = re
        .replace_all(&s, |c: &fancy_regex::Captures| {
            format!("{} have", c.get(1).map(|m| m.as_str()).unwrap_or(""))
        })
        .into_owned();
    let re = Regex::new(r"(?i)\b(\w+)'ll\b").unwrap();
    s = re
        .replace_all(&s, |c: &fancy_regex::Captures| {
            format!("{} will", c.get(1).map(|m| m.as_str()).unwrap_or(""))
        })
        .into_owned();
    let re = Regex::new(r"(?i)\b(\w+)'d\b").unwrap();
    s = re
        .replace_all(&s, |c: &fancy_regex::Captures| {
            format!("{} would", c.get(1).map(|m| m.as_str()).unwrap_or(""))
        })
        .into_owned();
    let re = Regex::new(r"(?i)\b(\w+)'m\b").unwrap();
    s = re
        .replace_all(&s, |c: &fancy_regex::Captures| {
            format!("{} am", c.get(1).map(|m| m.as_str()).unwrap_or(""))
        })
        .into_owned();
    let re_its = Regex::new(r"(?i)\bit's\b").unwrap();
    s = re_its.replace_all(&s, "it is").into_owned();
    s
}

pub fn remove_html_tags(text: &str) -> String {
    RE_HTML.replace_all(text, " ").into_owned()
}

pub fn remove_urls(text: &str) -> String {
    RE_URL.replace_all(text, "").trim().to_string()
}

pub fn remove_emails(text: &str) -> String {
    RE_EMAIL.replace_all(text, "").trim().to_string()
}

pub fn remove_hashtags(text: &str) -> String {
    RE_HASHTAG.replace_all(text, "").into_owned()
}

pub fn remove_mentions(text: &str) -> String {
    RE_MENTION.replace_all(text, "").into_owned()
}

pub fn remove_punctuation(text: &str) -> String {
    RE_PUNCT.replace_all(text, " ").into_owned()
}

pub fn remove_extra_whitespace(text: &str) -> String {
    RE_SPACES.replace_all(text.trim(), " ").into_owned()
}

pub fn normalize_unicode_nfc(text: &str) -> String {
    text.nfc().collect()
}

pub fn remove_accents(text: &str) -> String {
    text.nfd().filter(|c| !is_combining_mark(*c)).collect()
}

pub fn remove_stopwords(text: &str, stopwords: Option<&HashSet<String>>) -> String {
    let default: HashSet<String> = [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might", "this", "that", "these",
        "those", "it", "its", "i", "me", "my", "we", "our", "you", "your", "he", "she", "him",
        "her", "they", "them", "their",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();
    let sw = stopwords.unwrap_or(&default);
    text.split_whitespace()
        .filter(|t| !sw.contains(&t.to_lowercase()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Matches Python `TextPreprocessor(remove_punctuation=False)` used by KittenTTS ONNX.
#[derive(Clone)]
pub struct TextPreprocessor {
    pub lowercase: bool,
    pub replace_numbers: bool,
    pub replace_floats: bool,
    pub expand_contractions: bool,
    pub expand_model_names: bool,
    pub expand_ordinals: bool,
    pub expand_percentages: bool,
    pub expand_currency: bool,
    pub expand_time: bool,
    pub expand_ranges: bool,
    pub expand_units: bool,
    pub expand_scale_suffixes: bool,
    pub expand_scientific_notation: bool,
    pub expand_fractions: bool,
    pub expand_decades: bool,
    pub expand_phone_numbers: bool,
    pub expand_ip_addresses: bool,
    pub normalize_leading_decimals: bool,
    pub expand_roman_numerals: bool,
    pub remove_urls: bool,
    pub remove_emails: bool,
    pub remove_html: bool,
    pub remove_hashtags: bool,
    pub remove_mentions: bool,
    pub remove_punctuation: bool,
    pub remove_stopwords: bool,
    pub stopwords: Option<HashSet<String>>,
    pub normalize_unicode: bool,
    pub remove_accents: bool,
    pub remove_extra_whitespace: bool,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextPreprocessor {
    pub fn new() -> Self {
        Self {
            lowercase: true,
            replace_numbers: true,
            replace_floats: true,
            expand_contractions: true,
            expand_model_names: true,
            expand_ordinals: true,
            expand_percentages: true,
            expand_currency: true,
            expand_time: true,
            expand_ranges: true,
            expand_units: true,
            expand_scale_suffixes: true,
            expand_scientific_notation: true,
            expand_fractions: true,
            expand_decades: true,
            expand_phone_numbers: true,
            expand_ip_addresses: true,
            normalize_leading_decimals: true,
            expand_roman_numerals: false,
            remove_urls: true,
            remove_emails: true,
            remove_html: true,
            remove_hashtags: false,
            remove_mentions: false,
            remove_punctuation: true,
            remove_stopwords: false,
            stopwords: None,
            normalize_unicode: true,
            remove_accents: false,
            remove_extra_whitespace: true,
        }
    }

    /// Same defaults as Python but keep prosodic punctuation for TTS.
    pub fn kitten_onnx_default() -> Self {
        let mut p = Self::new();
        p.remove_punctuation = false;
        p
    }

    pub fn process(&self, text: &str) -> String {
        let mut text = text.to_string();

        if self.normalize_unicode {
            text = normalize_unicode_nfc(&text);
        }
        if self.remove_html {
            text = remove_html_tags(&text);
        }
        if self.remove_urls {
            text = remove_urls(&text);
        }
        if self.remove_emails {
            text = remove_emails(&text);
        }
        if self.remove_hashtags {
            text = remove_hashtags(&text);
        }
        if self.remove_mentions {
            text = remove_mentions(&text);
        }
        if self.expand_contractions {
            text = expand_contractions(&text);
        }
        if self.expand_ip_addresses {
            text = expand_ip_addresses(&text);
        }
        if self.normalize_leading_decimals {
            text = normalize_leading_decimals(&text);
        }
        if self.expand_currency {
            text = expand_currency(&text);
        }
        if self.expand_percentages {
            text = expand_percentages(&text);
        }
        if self.expand_scientific_notation {
            text = expand_scientific_notation(&text);
        }
        if self.expand_time {
            text = expand_time(&text);
        }
        if self.expand_ordinals {
            text = expand_ordinals(&text);
        }
        if self.expand_units {
            text = expand_units(&text);
        }
        if self.expand_scale_suffixes {
            text = expand_scale_suffixes(&text);
        }
        if self.expand_fractions {
            text = expand_fractions(&text);
        }
        if self.expand_decades {
            text = expand_decades(&text);
        }
        if self.expand_phone_numbers {
            text = expand_phone_numbers(&text);
        }
        if self.expand_ranges {
            text = expand_ranges(&text);
        }
        if self.expand_model_names {
            text = expand_model_names(&text);
        }
        if self.expand_roman_numerals {
            text = expand_roman_numerals(&text);
        }
        if self.replace_numbers {
            text = replace_numbers(&text, self.replace_floats);
        }
        if self.remove_accents {
            text = remove_accents(&text);
        }
        if self.remove_punctuation {
            text = remove_punctuation(&text);
        }
        if self.lowercase {
            text = text.to_lowercase();
        }
        if self.remove_stopwords {
            text = remove_stopwords(&text, self.stopwords.as_ref());
        }
        if self.remove_extra_whitespace {
            text = remove_extra_whitespace(&text);
        }

        text
    }
}
