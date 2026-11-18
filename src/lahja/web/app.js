const CLASSES = ["gulf", "levant", "egypt", "maghreb", "msa"];
const I18N = {
  en: {
    classes: { gulf: "Gulf", levant: "Levant", egypt: "Egypt", maghreb: "Maghreb", msa: "MSA" },
    placeholder: "Type a sentence in Arabic",
    scope: "Most reliable between 3 and 60 words.",
    noArabic: "Arabic script only.",
    failed: "No response from the model.",
    card: "Model card",
    cardHref: "https://github.com/pinpin-http/lahja/blob/main/docs/model_card.md",
  },
  fr: {
    classes: { gulf: "Golfe", levant: "Levant", egypt: "Égypte", maghreb: "Maghreb", msa: "Arabe standard" },
    placeholder: "Écrivez une phrase en arabe",
    scope: "Plus fiable entre 3 et 60 mots.",
    noArabic: "Écriture arabe uniquement.",
    failed: "Le modèle ne répond pas.",
    card: "Fiche du modèle",
    cardHref: "https://github.com/pinpin-http/lahja/blob/main/docs/model_card.fr.md",
  },
};

const $ = (id) => document.getElementById(id);
const text = $("text");
let lang = "en";
let examples = {};
const cursor = {};
let last = null;
let timer = null;
let inflight = null;

function store(key, value) {
  try { value === undefined ? localStorage.getItem(key) : localStorage.setItem(key, value); } catch (_) {}
}
function load(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}

function setLang(l) {
  lang = l;
  store("lahja-lang", l);
  document.documentElement.lang = l;
  const t = I18N[l];
  text.placeholder = t.placeholder;
  $("card").textContent = t.card;
  $("card").href = t.cardHref;
  document.querySelectorAll(".lang button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === l)));
  renderExamples();
  if (last) render(last);
}

function renderExamples() {
  const nav = $("examples");
  nav.replaceChildren();
  for (const c of CLASSES) {
    if (!examples[c] || !examples[c].length) continue;
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = I18N[lang].classes[c];
    b.addEventListener("click", () => {
      cursor[c] = ((cursor[c] ?? -1) + 1) % examples[c].length;
      text.value = examples[c][cursor[c]];
      run();
    });
    nav.append(b);
  }
}

function note(msg) {
  $("note").textContent = msg || "";
  $("note").hidden = !msg;
}

function render(res) {
  const t = I18N[lang];
  const p = res.probabilities;
  $("label").textContent = t.classes[res.label];
  $("conf").textContent = Math.round(p[res.label] * 100) + " %";
  const bars = $("bars");
  bars.replaceChildren();
  for (const c of CLASSES) {
    const li = document.createElement("li");
    if (c === res.label) li.className = "win";
    const pct = Math.round(p[c] * 100);
    li.innerHTML = `<span class="name"></span><span class="track"><span class="fill"></span></span><span class="pct"></span>`;
    li.querySelector(".name").textContent = t.classes[c];
    li.querySelector(".pct").textContent = pct + " %";
    li.setAttribute("aria-label", `${t.classes[c]} ${pct} %`);
    bars.append(li);
    requestAnimationFrame(() => (li.querySelector(".fill").style.width = Math.max(p[c] * 100, 0.5) + "%"));
  }
  const words = $("words");
  words.replaceChildren();
  const ws = res.words || [];
  const max = Math.max(0, ...ws.map((w) => w.weight));
  ws.forEach((w, i) => {
    if (i) words.append(" ");
    if (max > 0 && w.weight > 0.02 * Math.max(1, max)) {
      const m = document.createElement("mark");
      m.textContent = w.word;
      m.style.background = `rgba(var(--hl), ${(0.12 + 0.5 * (w.weight / max)).toFixed(2)})`;
      m.title = "+" + Math.round(w.weight * 100) + " %";
      words.append(m);
    } else {
      words.append(w.word);
    }
  });
  $("result").hidden = false;
  note(res.in_scope ? "" : t.scope);
}

async function run() {
  clearTimeout(timer);
  const value = text.value.trim();
  if (!value) {
    $("result").hidden = true;
    note("");
    last = null;
    return;
  }
  if (inflight) inflight.abort();
  inflight = new AbortController();
  try {
    const r = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: value, explain: true }),
      signal: inflight.signal,
    });
    if (r.status === 422) {
      $("result").hidden = true;
      last = null;
      note(I18N[lang].noArabic);
      return;
    }
    if (!r.ok) throw new Error(r.status);
    last = await r.json();
    render(last);
  } catch (e) {
    if (e.name !== "AbortError") note(I18N[lang].failed);
  }
}

text.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(run, 250);
});
document.querySelectorAll(".lang button").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));

fetch("/static/examples.json")
  .then((r) => (r.ok ? r.json() : {}))
  .then((e) => { examples = e; renderExamples(); })
  .catch(() => {});

setLang(load("lahja-lang") || ((navigator.language || "").toLowerCase().startsWith("fr") ? "fr" : "en"));
