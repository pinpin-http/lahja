import { DialectModel, NoArabicText } from "./engine.js";

// Public page: the browser downloads the model from MODEL and runs it itself.
// Local `make serve`: MODEL is empty and the page calls the API that serves it.
const MODEL = (document.querySelector('meta[name="lahja-model"]')?.content || "").replace(/\/$/, "");
const MODE = MODEL ? "browser" : "api";
const CLASSES = ["gulf", "levant", "egypt", "maghreb", "msa"];
const HUES = { gulf: "#f25ca8", levant: "#9b72ff", egypt: "#4f8bff", maghreb: "#34c3ff", msa: "#ffb86b" };
const NBSP = " ";

const I18N = {
  en: {
    classes: { gulf: "Gulf", levant: "Levant", egypt: "Egypt", maghreb: "Maghreb", msa: "Standard Arabic" },
    lede: "Type a sentence in Arabic. Lahja tells you whether it sounds like the Gulf, the Levant, Egypt, the Maghreb or Modern Standard Arabic.",
    placeholder: "Type or paste a sentence in Arabic",
    examples: "Try",
    detected: "Detected dialect",
    weighed: "Words that weighed most",
    spread: "Several words point to this dialect: none of them weighs much on its own.",
    explaining: "Measuring each word…",
    scope: "Most reliable between 3 and 60 words.",
    noArabic: "Arabic script only.",
    failed: "The model is not answering. Try again in a moment.",
    loadFailed: "The model could not be loaded. Reload the page to try again.",
    waking: "Waiting for the model…",
    downloading: (pct) => `Downloading the model (156${NBSP}MB, only the first time)… ${pct}${NBSP}%`,
    preparing: "Preparing the model…",
    status: { ready: "Model ready", waking: "Connecting…", loading: "Loading model", offline: "Unavailable" },
    words: (n) => `${n} word${n === 1 ? "" : "s"}`,
    clear: "Clear",
    analyse: "Analyse",
    card: "Model card",
    cardHref: "https://huggingface.co/pinpin-http/lahja",
    hub: "Hugging Face",
  },
  fr: {
    classes: { gulf: "Golfe", levant: "Levant", egypt: "Égypte", maghreb: "Maghreb", msa: "Arabe standard" },
    lede: "Écrivez une phrase en arabe. Lahja dit si elle sonne comme le Golfe, le Levant, l'Égypte, le Maghreb ou l'arabe standard.",
    placeholder: "Écrivez ou collez une phrase en arabe",
    examples: "Essayer",
    detected: "Dialecte détecté",
    weighed: "Les mots qui ont le plus pesé",
    spread: "Plusieurs mots désignent ce dialecte : aucun ne pèse beaucoup à lui seul.",
    explaining: "Mesure de chaque mot…",
    scope: "Plus fiable entre 3 et 60 mots.",
    noArabic: "Écriture arabe uniquement.",
    failed: "Le modèle ne répond pas. Réessayez dans un instant.",
    loadFailed: "Le modèle n'a pas pu être chargé. Rechargez la page pour réessayer.",
    waking: "En attente du modèle…",
    downloading: (pct) => `Téléchargement du modèle (156${NBSP}Mo, la première fois seulement)… ${pct}${NBSP}%`,
    preparing: "Préparation du modèle…",
    status: { ready: "Modèle prêt", waking: "Connexion…", loading: "Chargement", offline: "Indisponible" },
    words: (n) => `${n}${NBSP}mot${n > 1 ? "s" : ""}`,
    clear: "Effacer",
    analyse: "Analyser",
    card: "Fiche du modèle",
    cardHref: "https://huggingface.co/pinpin-http/lahja/blob/main/model_card.fr.md",
    hub: "Hugging Face",
  },
};

const $ = (id) => document.getElementById(id);
const text = $("text");
const composer = $("composer");
const RING = 2 * Math.PI * 27;

let lang = "en";
let examples = {};
const cursor = {};
let last = null;
let state = "waking";
let progress = null; // 0..100 while the model downloads
let backend = null; // { predict(text, explain) } once the model can answer
let timer = null;
let seq = 0;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function saved(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (_) {}
  return null;
}

/* ---------- language ---------- */
function setLang(l) {
  lang = l;
  saved("lahja-lang", l);
  const t = I18N[l];
  document.documentElement.lang = l;
  document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = t[el.dataset.i18n]));
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t[el.dataset.i18nAria]));
  document.querySelectorAll(".lang button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === l)));
  text.placeholder = t.placeholder;
  $("card").href = t.cardHref;
  setStatus(state);
  renderExamples();
  updateCount();
  if (last) render(last);
  else if (text.value.trim() && !backend) waiting();
}

/* ---------- status ---------- */
function setStatus(s) {
  state = s;
  const t = I18N[lang];
  $("status").dataset.state = s;
  $("status-text").textContent = t.status[s] + (s === "loading" && progress !== null ? ` ${progress}${NBSP}%` : "");
}

function waiting() {
  const t = I18N[lang];
  if (state === "offline") return pending(null, MODE === "browser" ? t.loadFailed : t.failed);
  if (MODE === "browser") pending(progress, progress === null || progress >= 100 ? t.preparing : t.downloading(progress));
  else pending(null, t.waking);
}

/* ---------- backends ---------- */
function apiBackend() {
  return {
    async predict(value, explain) {
      const r = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: value, explain }),
      });
      if (r.status === 422) throw new NoArabicText(value);
      if (!r.ok) throw new Error(String(r.status));
      return r.json();
    },
  };
}

async function startApi() {
  setStatus("waking");
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch("/health", { cache: "no-store" });
      if (r.ok) {
        backend = apiBackend();
        return ready();
      }
    } catch (_) {}
    await sleep(3000);
  }
  failed();
}

const CACHE = "lahja-" + (MODEL.match(/resolve\/([0-9a-f]{7})/)?.[1] || "model");

async function download(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) {
      onProgress?.(1, 1);
      return new Uint8Array(await hit.arrayBuffer());
    }
  } catch (_) {
    cache = null;
  }
  const r = await fetch(url, { mode: "cors" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const total = Number(r.headers.get("content-length")) || 0;
  let buf;
  if (r.body && total) {
    buf = new Uint8Array(total);
    const reader = r.body.getReader();
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf.set(value, got);
      got += value.length;
      onProgress?.(got, total);
    }
  } else {
    buf = new Uint8Array(await r.arrayBuffer());
  }
  try {
    await cache?.put(url, new Response(buf, { headers: { "Content-Type": "application/octet-stream" } }));
  } catch (_) {}
  return buf;
}

async function startBrowser() {
  progress = 0;
  setStatus("loading");
  try {
    caches.keys().then((ks) => ks.filter((k) => k.startsWith("lahja-") && k !== CACHE).forEach((k) => caches.delete(k)));
  } catch (_) {}
  try {
    const ort = await import("/static/ort/ort.wasm.min.mjs");
    ort.env.wasm.proxy = true; // inference in a worker: the page stays fluid
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    const [cfg, tok, onnx] = await Promise.all([
      download(MODEL + "/lahja.json").then((b) => JSON.parse(new TextDecoder().decode(b))),
      download(MODEL + "/tokenizer.json").then((b) => JSON.parse(new TextDecoder().decode(b))),
      download(MODEL + "/model.int8.onnx", (got, total) => {
        const pct = Math.floor((100 * got) / total);
        if (pct !== progress) {
          progress = pct;
          setStatus("loading");
          if (text.value.trim() && !backend) waiting();
        }
      }),
    ]);
    progress = 100;
    if (text.value.trim()) waiting();
    const session = await ort.InferenceSession.create(onnx, { executionProviders: ["wasm"] });
    const model = new DialectModel(ort, session, tok, cfg);
    await model.predict("هذا اختبار قصير"); // warm-up
    backend = { predict: (value, explain) => (explain ? model.explain(value) : model.predict(value)) };
    ready();
  } catch (e) {
    console.error(e);
    failed();
  }
}

function ready() {
  progress = null;
  setStatus("ready");
  if (text.value.trim()) run();
}

function failed() {
  setStatus("offline");
  if (text.value.trim()) waiting();
}

/* ---------- examples ---------- */
function renderExamples() {
  const box = $("examples");
  box.replaceChildren();
  for (const c of CLASSES) {
    if (!examples[c] || !examples[c].length) continue;
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.style.setProperty("--hue", HUES[c]);
    b.textContent = I18N[lang].classes[c];
    b.addEventListener("click", () => {
      cursor[c] = ((cursor[c] ?? -1) + 1) % examples[c].length;
      text.value = examples[c][cursor[c]];
      grow();
      run();
    });
    box.append(b);
  }
}

/* ---------- rendering ---------- */
function note(msg) {
  $("note").textContent = msg || "";
  $("note").hidden = !msg;
}

function pending(pct, msg) {
  $("result").hidden = true;
  $("pending").hidden = false;
  $("progress").hidden = pct === null || pct === undefined;
  $("progress-fill").style.width = (pct || 0) + "%";
  $("pending-text").textContent = msg || "";
  $("pending-text").hidden = !msg;
}

function hidePending() {
  $("pending").hidden = true;
}

function render(res) {
  const t = I18N[lang];
  const p = res.probabilities;
  document.body.style.setProperty("--tint", HUES[res.label]);
  $("label").textContent = t.classes[res.label];
  $("conf").textContent = Math.round(p[res.label] * 100) + NBSP + "%";
  $("ring-fill").style.strokeDashoffset = String(RING * (1 - p[res.label]));

  const bars = $("bars");
  bars.replaceChildren();
  for (const c of [...CLASSES].sort((a, b) => p[b] - p[a])) {
    const pct = Math.round(p[c] * 100);
    const li = document.createElement("li");
    if (c === res.label) li.className = "win";
    li.style.setProperty("--hue", HUES[c]);
    li.setAttribute("aria-label", `${t.classes[c]} ${pct} %`);
    const name = document.createElement("span");
    name.textContent = t.classes[c];
    const track = document.createElement("span");
    track.className = "track";
    const fill = document.createElement("span");
    fill.className = "fill";
    track.append(fill);
    const val = document.createElement("span");
    val.className = "pct";
    val.textContent = pct + NBSP + "%";
    li.append(name, track, val);
    bars.append(li);
    requestAnimationFrame(() => requestAnimationFrame(() => (fill.style.width = Math.max(p[c] * 100, 0.6) + "%")));
  }
  renderWords(res);
  hidePending();
  $("result").hidden = false;
  note(res.in_scope ? "" : t.scope);
}

function renderWords(res) {
  const t = I18N[lang];
  const box = $("words");
  const hint = $("words-hint");
  box.replaceChildren();
  if (!res.words) {
    box.textContent = text.value.trim();
    hint.textContent = t.explaining;
    hint.hidden = false;
    return;
  }
  const max = Math.max(0, ...res.words.map((w) => w.weight));
  res.words.forEach((w, i) => {
    if (i) box.append(" ");
    if (max >= 0.02 && w.weight >= Math.max(0.01, 0.15 * max)) {
      const m = document.createElement("mark");
      m.textContent = w.word;
      m.style.setProperty("--w", (0.35 + 0.65 * (w.weight / max)).toFixed(2));
      m.title = "+" + Math.round(w.weight * 100) + " %";
      box.append(m);
    } else {
      box.append(w.word);
    }
  });
  hint.textContent = max < 0.02 ? t.spread : "";
  hint.hidden = max >= 0.02;
}

/* ---------- prediction ---------- */
async function run() {
  clearTimeout(timer);
  const value = text.value.trim();
  const id = ++seq;
  updateCount();
  if (!value) {
    $("result").hidden = true;
    hidePending();
    note("");
    last = null;
    document.body.style.removeProperty("--tint");
    return;
  }
  if (!backend) return waiting();
  composer.classList.add("busy");
  if ($("result").hidden) pending(null, "");
  try {
    const fast = await backend.predict(value, false);
    if (id !== seq) return;
    last = fast;
    render(last);
    const full = await backend.predict(value, true);
    if (id !== seq) return;
    last = full;
    renderWords(last);
  } catch (e) {
    if (id !== seq) return;
    last = null;
    $("result").hidden = true;
    hidePending();
    note(e instanceof NoArabicText ? I18N[lang].noArabic : I18N[lang].failed);
  } finally {
    if (id === seq) composer.classList.remove("busy");
  }
}

/* ---------- input ---------- */
function updateCount() {
  const n = text.value.trim() ? text.value.trim().split(/\s+/).length : 0;
  $("count").textContent = n ? I18N[lang].words(n) : "";
  $("clear").hidden = !n;
}
function grow() {
  text.style.height = "auto";
  text.style.height = text.scrollHeight + "px";
}

text.addEventListener("input", () => {
  grow();
  updateCount();
  clearTimeout(timer);
  timer = setTimeout(run, MODE === "browser" ? 250 : 380);
});
text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    run();
  }
});
$("go").addEventListener("click", () => {
  run();
  text.focus();
});
$("clear").addEventListener("click", () => {
  text.value = "";
  grow();
  run();
  text.focus();
});
document.querySelectorAll(".lang button").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));

fetch("/static/examples.json")
  .then((r) => (r.ok ? r.json() : {}))
  .then((e) => {
    examples = e;
    renderExamples();
  })
  .catch(() => {});

/* ---------- background follows the pointer a little ---------- */
const parallax = document.querySelector(".parallax");
if (parallax && matchMedia("(pointer: fine)").matches && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
  let raf = 0;
  addEventListener("pointermove", (e) => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      parallax.style.setProperty("--px", ((e.clientX / innerWidth - 0.5) * 6).toFixed(2) + "vw");
      parallax.style.setProperty("--py", ((e.clientY / innerHeight - 0.5) * 6).toFixed(2) + "vh");
    });
  });
}

const params = new URLSearchParams(location.search);
const q = params.get("q");
if (q) {
  text.value = q;
  grow();
}
setLang(
  (I18N[params.get("lang")] && params.get("lang")) ||
    saved("lahja-lang") ||
    ((navigator.language || "").toLowerCase().startsWith("fr") ? "fr" : "en"),
);
if (matchMedia("(pointer: fine)").matches) text.focus();
if (MODE === "browser") startBrowser();
else startApi();
