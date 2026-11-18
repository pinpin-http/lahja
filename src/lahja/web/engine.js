// Lahja in the browser: the normalisation, tokenisation, model call and word influence of the
// Python package (lahja.normalize, tokenizers' BertWordPiece, lahja.inference), ported line by
// line. tests/test_web_engine.py checks that both give the same tokens and the same logits.

const MIN_WORDS = 3;
const MAX_WORDS = 60;

// Python's str.isspace() set, so that \s means the same thing on both sides
const WS = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const URL_RE = new RegExp(`(https?://[^${WS}]+|www\\.[^${WS}]+|[^${WS}]+@[^${WS}]+\\.[^${WS}]+|@[\\p{L}\\p{N}_]+)`, "gu");
const SPACES = new RegExp(`[${WS}]+`, "gu");
const DIACRITICS = /[ؐ-ًؚ-ٰٟۖ-ۭـ]/gu;
const MAP = {
  "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", // alif variants
  "ة": "ه", // ta marbuta
  "ى": "ي", "ی": "ي", "ې": "ي", // alif maqsura, Persian ya
  "ک": "ك", "ڪ": "ك", // Persian kaf
  "ھ": "ه", "ۀ": "ه", // heh doachashmee, heh with yeh
};
const MAP_RE = new RegExp(`[${Object.keys(MAP).join("")}]`, "gu");
const NOT_ARABIC = /[^ء-غف-يپچژڤڨڭگ ]+/gu;
const REPEAT = /(.)\1{2,}/gu;

export function normalize(text) {
  if (!text) return "";
  let t = text.normalize("NFKC");
  t = t.replace(URL_RE, " ");
  t = t.replace(DIACRITICS, "");
  t = t.replace(MAP_RE, (c) => MAP[c]);
  t = t.replace(SPACES, " ");
  t = t.replace(NOT_ARABIC, " ");
  t = t.replace(REPEAT, "$1$1");
  return t.replace(SPACES, " ").replace(new RegExp(`^[${WS}]+|[${WS}]+$`, "gu"), "");
}

const splitWords = (s) => s.split(SPACES).filter(Boolean);

/** BertNormalizer + BertPreTokenizer + WordPiece + [CLS] … [SEP], from a tokenizer.json. */
export class WordPiece {
  constructor(json, maxLen = 128) {
    const m = json.model;
    const norm = json.normalizer || {};
    this.vocab = new Map(Object.entries(m.vocab));
    this.prefix = m.continuing_subword_prefix ?? "##";
    this.maxChars = m.max_input_chars_per_word ?? 100;
    this.unk = this.vocab.get(m.unk_token);
    this.cls = this.vocab.get("[CLS]");
    this.sep = this.vocab.get("[SEP]");
    this.pad = this.vocab.get("[PAD]") ?? 0;
    this.lowercase = norm.lowercase ?? true;
    this.stripAccents = norm.strip_accents ?? this.lowercase;
    this.maxLen = maxLen;
  }

  normalizeBert(s) {
    // clean_text: drop control characters, map whitespace to a plain space
    s = s.replace(/[\0�]|(?![\t\n\r])\p{C}/gu, "").replace(/\s/gu, " ");
    if (this.stripAccents) s = s.normalize("NFD").replace(/\p{Mn}/gu, "");
    if (this.lowercase) s = s.toLowerCase();
    return s;
  }

  preTokenize(s) {
    // split on whitespace, then isolate every punctuation character
    return s.split(/\s+/u).filter(Boolean).flatMap((w) => w.split(/([\p{P}!-\/:-@\[-`{-~])/u).filter(Boolean));
  }

  wordPiece(word) {
    const chars = Array.from(word);
    if (chars.length > this.maxChars) return [this.unk];
    const out = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let id;
      while (start < end) {
        const sub = (start > 0 ? this.prefix : "") + chars.slice(start, end).join("");
        id = this.vocab.get(sub);
        if (id !== undefined) break;
        end--;
      }
      if (id === undefined) return [this.unk];
      out.push(id);
      start = end;
    }
    return out;
  }

  encode(text) {
    const ids = this.preTokenize(this.normalizeBert(text)).flatMap((w) => this.wordPiece(w));
    return [this.cls, ...ids.slice(0, this.maxLen - 2), this.sep];
  }
}

function softmax(row) {
  const m = Math.max(...row);
  const e = row.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

export class NoArabicText extends Error {}

export class DialectModel {
  /** ort: the onnxruntime-web module; session: an InferenceSession; config: lahja.json. */
  constructor(ort, session, tokenizerJson, config) {
    this.ort = ort;
    this.session = session;
    this.config = config;
    this.classes = config.classes;
    this.temperature = Number(config.temperature ?? 1);
    this.tok = new WordPiece(tokenizerJson, Number(config.max_len ?? 128));
  }

  async logits(normalized) {
    const enc = normalized.map((s) => this.tok.encode(s));
    const L = Math.max(...enc.map((e) => e.length));
    const ids = new BigInt64Array(enc.length * L).fill(BigInt(this.tok.pad));
    const mask = new BigInt64Array(enc.length * L);
    enc.forEach((e, i) => {
      e.forEach((id, j) => {
        ids[i * L + j] = BigInt(id);
        mask[i * L + j] = 1n;
      });
    });
    const T = this.ort.Tensor;
    const out = await this.session.run({
      input_ids: new T("int64", ids, [enc.length, L]),
      attention_mask: new T("int64", mask, [enc.length, L]),
    });
    const flat = out.logits.data;
    const k = this.classes.length;
    return enc.map((_, i) => Array.from(flat.slice(i * k, (i + 1) * k)));
  }

  async proba(normalized) {
    return (await this.logits(normalized)).map((row) => softmax(row.map((x) => x / this.temperature)));
  }

  result(norm, p) {
    const best = p.indexOf(Math.max(...p));
    const n = splitWords(norm).length;
    return {
      label: this.classes[best],
      probabilities: Object.fromEntries(this.classes.map((c, i) => [c, Math.round(p[i] * 1e4) / 1e4])),
      n_words: n,
      in_scope: n >= MIN_WORDS && n <= MAX_WORDS,
    };
  }

  async predict(text) {
    const norm = normalize(text);
    if (!norm) throw new NoArabicText(text);
    const [p] = await this.proba([norm]);
    return { ...this.result(norm, p), words: null };
  }

  /** Prediction plus, for each word, how much the predicted class loses when it is removed. */
  async explain(text, maxWords = 60) {
    const norm = normalize(text);
    if (!norm) throw new NoArabicText(text);
    let raw = splitWords(text);
    let owner = raw.flatMap((tok, i) => splitWords(normalize(tok)).map(() => i));
    const words = splitWords(norm);
    if (owner.length !== words.length) {
      raw = words;
      owner = words.map((_, i) => i);
    }
    const n = Math.min(words.length, maxWords);
    const variants = [norm];
    for (let j = 0; j < n; j++) variants.push([...words.slice(0, j), ...words.slice(j + 1)].join(" ") || norm);
    const probs = await this.proba(variants);
    const base = probs[0];
    const k = base.indexOf(Math.max(...base));
    const drop = probs.slice(1).map((p) => (words.length === 1 ? base[k] : base[k] - p[k]));
    const weights = new Array(raw.length).fill(0);
    for (let j = 0; j < n; j++) weights[owner[j]] += drop[j];
    return {
      ...this.result(norm, base),
      words: raw.map((w, i) => ({ word: w, weight: Math.round(weights[i] * 1e4) / 1e4 })),
    };
  }
}
