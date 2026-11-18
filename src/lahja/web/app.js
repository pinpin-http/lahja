const API = (document.querySelector('meta[name="lahja-api"]')?.content || "").replace(/\/$/, "");
const CLASSES = ["gulf", "levant", "egypt", "maghreb", "msa"];
const HUES = { gulf: "#f25ca8", levant: "#9b72ff", egypt: "#4f8bff", maghreb: "#34c3ff", msa: "#ffb86b" };
const NBSP = " ";

const I18N = {
  en: {
    classes: { gulf: "Gulf", levant: "Levant", egypt: "Egypt", maghreb: "Maghreb", msa: "Standard Arabic" },
    eyebrow: "Arabic dialect identification",
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
    waking: "The model is waking up on Hugging Face. This can take up to a minute.",
    status: { ready: "Model ready", waking: "Waking up…", offline: "Offline" },
    words: (n) => `${n} word${n === 1 ? "" : "s"}`,
    clear: "Clear",
    analyse: "Analyse",
    statF1: `80.4${NBSP}%`,
    statF1Label: "macro-F1 on 2,749 test sentences",
    statOod: `78.5${NBSP}%`,
    statOodLabel: "macro-F1 on 3,000 tweets never seen in training",
    statLatency: `24${NBSP}ms`,
    statLatencyLabel: "per sentence on 2 CPU threads (95th percentile)",
    about: "A MARBERT encoder fine-tuned on three openly licensed corpora (FLORES-200, SADA, Lahga), quantised to 8-bit integers and served by ONNX Runtime on CPU. Highlighted words are the ones whose removal lowers the prediction most.",
    limits: "Gulf mostly means Saudi Arabia. Iraqi, Yemeni and Sudanese Arabic are not covered, nor is Arabizi. Below three words, the answer is close to a guess.",
    card: "Model card",
    cardHref: "https://huggingface.co/pinpin-http/lahja",
  },
  fr: {
    classes: { gulf: "Golfe", levant: "Levant", egypt: "Égypte", maghreb: "Maghreb", msa: "Arabe standard" },
    eyebrow: "Identification des dialectes arabes",
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
    waking: "Le modèle se réveille sur Hugging Face, ça peut prendre une minute.",
    status: { ready: "Modèle prêt", waking: "Réveil…", offline: "Hors ligne" },
    words: (n) => `${n}${NBSP}mot${n > 1 ? "s" : ""}`,
    clear: "Effacer",
    analyse: "Analyser",
    statF1: `80,4${NBSP}%`,
    statF1Label: "de macro-F1 sur 2 749 phrases de test",
    statOod: `78,5${NBSP}%`,
    statOodLabel: "sur 3 000 tweets jamais vus à l'entraînement",
    statLatency: `24${NBSP}ms`,
    statLatencyLabel: "par phrase sur 2 threads CPU (95e centile)",
    about: "Un encodeur MARBERT affiné sur trois corpus sous licence ouverte (FLORES-200, SADA, Lahga), quantifié en entiers 8 bits et servi par ONNX Runtime sur CPU. Les mots surlignés sont ceux dont le retrait fait le plus baisser la prédiction.",
    limits: "Golfe veut surtout dire Arabie saoudite. L'irakien, le yéménite, le soudanais et l'arabizi ne sont pas couverts. En dessous de trois mots, la réponse tient presque du hasard.",
    card: "Fiche du modèle",
    cardHref: "https://huggingface.co/pinpin-http/lahja/blob/main/model_card.fr.md",
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
let ready = false;
let state = "waking";
let timer = null;
let inflight = null;

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
}

/* ---------- model status (a free Space sleeps when nobody uses it) ---------- */
function setStatus(s) {
  state = s;
  $("status").dataset.state = s;
  $("status-text").textContent = I18N[lang].status[s];
}

async function wake() {
  setStatus("waking");
  const start = Date.now();
  while (Date.now() - start < 4 * 60 * 1000) {
    try {
      const r = await fetch(API + "/health", { cache: "no-store", signal: AbortSignal.timeout(10000) });
      if (r.ok && (await r.json()).status === "ok") {
        ready = true;
        setStatus("ready");
        if (text.value.trim()) run();
        return;
      }
    } catch (_) {}
    if (text.value.trim()) pending(I18N[lang].waking);
    await sleep(4000);
  }
  setStatus("offline");
  if (text.value.trim()) {
    pending(null);
    note(I18N[lang].failed);
  }
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
    b.append(document.createElement("i"), I18N[lang].classes[c]);
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

function pending(msg) {
  const show = msg !== null;
  $("pending").hidden = !show || !$("result").hidden;
  $("pending-text").textContent = msg || "";
  $("pending-text").hidden = !msg;
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
  $("pending").hidden = true;
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

/* ---------- requests ---------- */
async function post(value, explain, signal) {
  const r = await fetch(API + "/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: value, explain }),
    signal,
  });
  if (r.status === 422) return { status: 422 };
  if (!r.ok) throw new Error(String(r.status));
  return { status: 200, data: await r.json() };
}

async function run() {
  clearTimeout(timer);
  const value = text.value.trim();
  updateCount();
  if (inflight) inflight.abort();
  if (!value) {
    $("result").hidden = true;
    pending(null);
    note("");
    last = null;
    return;
  }
  if (!ready) {
    pending(I18N[lang].waking);
    return;
  }
  inflight = new AbortController();
  const { signal } = inflight;
  composer.classList.add("busy");
  if ($("result").hidden) pending("");
  try {
    const fast = await post(value, false, signal);
    if (fast.status === 422) {
      $("result").hidden = true;
      pending(null);
      last = null;
      note(I18N[lang].noArabic);
      return;
    }
    last = fast.data;
    render(last);
    const full = await post(value, true, signal);
    if (full.status === 200 && text.value.trim() === value) {
      last = full.data;
      renderWords(last);
    }
  } catch (e) {
    if (e.name === "AbortError") return;
    pending(null);
    note(I18N[lang].failed);
    ready = false;
    wake();
  } finally {
    if (!signal.aborted) composer.classList.remove("busy");
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
  timer = setTimeout(run, 380);
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

const params = new URLSearchParams(location.search);
setLang(
  (I18N[params.get("lang")] && params.get("lang")) ||
    saved("lahja-lang") ||
    ((navigator.language || "").toLowerCase().startsWith("fr") ? "fr" : "en"),
);
const q = params.get("q");
if (q) {
  text.value = q;
  grow();
  updateCount();
}
if (matchMedia("(pointer: fine)").matches) text.focus();
wake();
