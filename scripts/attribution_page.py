"""Render a token-attribution JSON (from token_attribution.py) as a self-contained HTML page.

    python scripts/attribution_page.py results/glm53/attribution_744ca2f7.json [OUT.html]
"""

import json
import sys
from pathlib import Path

TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: one reading column; summary + step overview on top, the shaded transcript below. */
:root {
  --bg: #fcfcfb; --panel: #f4f3f0; --fg: #1a1a19; --fg-2: #52514e; --rule: #e4e3df;
  --help: 42,120,214;   /* blue: true side prompt makes the token easier to predict */
  --hurt: 227,73,72;    /* red: harder to predict */
  --mid: #f0efec;
  --sans: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #1a1a19; --panel: #232321; --fg: #f0efec; --fg-2: #c3c2b7; --rule: #383835;
  --help: 57,135,229; --hurt: 230,103,103; --mid: #383835; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #1a1a19; --panel: #232321; --fg: #f0efec; --fg-2: #c3c2b7; --rule: #383835;
  --help: 57,135,229; --hurt: 230,103,103; --mid: #383835; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 14px; line-height: 1.5; }
.wrap { max-width: 1040px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 28px; }
h1 { font-size: 22px; font-weight: 600; margin: 0; text-wrap: balance; }
h2 { font-size: 13px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--fg-2); margin: 0 0 10px; }
.meta { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; font-size: 13px; margin-top: 10px; }
.meta dt { color: var(--fg-2); } .meta dd { margin: 0; font-family: var(--mono); font-size: 12.5px; min-width: 0; overflow-wrap: anywhere; }
details.side { background: var(--panel); border-radius: 6px; padding: 10px 14px; }
details.side summary { cursor: pointer; font-weight: 500; }
details.side code { font-family: var(--mono); font-size: 12.5px; }
details.side pre { white-space: pre-wrap; font-family: var(--mono); font-size: 12.5px; margin: 10px 0 0; }
.totals { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.tot { border-top: 2px solid var(--rule); padding-top: 8px; }
.tot .k { font-size: 12px; color: var(--fg-2); } .tot .v { font-family: var(--mono); font-size: 18px; font-variant-numeric: tabular-nums; }
.tot .d { font-size: 12px; color: var(--fg-2); font-variant-numeric: tabular-nums; }
.controls { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; justify-content: space-between; }
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.seg button { font: inherit; font-size: 13px; padding: 6px 12px; background: transparent; color: var(--fg); border: 0; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--fg); color: var(--bg); }
.seg button:focus-visible, .step:focus-visible { outline: 2px solid rgb(var(--help)); outline-offset: 2px; }
.show { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; font-size: 13px; border: 0; padding: 0; margin: 0; }
.show legend { float: left; color: var(--fg-2); margin-right: 4px; padding: 0; }
.show label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
.show input:focus-visible { outline: 2px solid rgb(var(--help)); outline-offset: 2px; }
.toks span.off { display: none; }
.scale { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--fg-2); }
.scale .bar { width: 180px; height: 10px; border-radius: 2px;
  background: linear-gradient(90deg, rgba(var(--hurt),.85), var(--mid), rgba(var(--help),.85)); }
.steps { display: grid; gap: 2px; overflow-x: auto; }
.step { display: grid; grid-template-columns: 28px minmax(140px, 220px) 70px 1fr 72px; gap: 10px; align-items: center;
  padding: 3px 6px; border-radius: 4px; cursor: pointer; font-size: 12.5px; background: transparent; border: 0; color: inherit; text-align: left; font-family: inherit; }
.step:hover { background: var(--panel); }
.step .i, .step .n, .step .s { font-family: var(--mono); font-variant-numeric: tabular-nums; color: var(--fg-2); }
.step .n, .step .s { text-align: right; } .step .s { color: var(--fg); }
.track { position: relative; height: 12px; }
.track .zero { position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: var(--fg-2); opacity: .5; }
.track .b { position: absolute; top: 0; bottom: 0; border-radius: 2px; }
.msg { border-top: 1px solid var(--rule); padding-top: 10px; scroll-margin-top: 16px; min-width: 0; }
.msg.flash { animation: flash 1.2s ease-out; }
@keyframes flash { from { background: var(--panel); } to { background: transparent; } }
@media (prefers-reduced-motion: reduce) { .msg.flash { animation: none; } }
.mh { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: baseline; font-size: 12.5px; margin-bottom: 6px; }
.mh .role { font-weight: 600; } .mh .kind { color: var(--fg-2); font-family: var(--mono); }
.mh .sum { margin-left: auto; font-family: var(--mono); font-variant-numeric: tabular-nums; }
.toks { font-family: var(--mono); font-size: 12.5px; line-height: 1.7; white-space: pre-wrap; overflow-wrap: anywhere; }
.toks span { border-radius: 2px; }
.msg.tool .toks { max-height: 420px; overflow-y: auto; padding-right: 6px; }
.tip { position: fixed; pointer-events: none; z-index: 20; background: var(--fg); color: var(--bg); font-family: var(--mono);
  font-size: 12px; padding: 6px 8px; border-radius: 4px; white-space: pre; display: none; }
.note { font-size: 13px; color: var(--fg-2); max-width: 70ch; margin: 0; }
@media (max-width: 560px) {
  .step { grid-template-columns: 24px 1fr 64px; } .step .n, .step .track { display: none; }
  .totals { grid-template-columns: 1fr; }
}
</style>

<div class="wrap">
  <header>
    <h1>Where the true side prompt helps T predict U's transcript</h1>
    <dl class="meta" id="meta"></dl>
  </header>
  <details class="side"><summary>Hidden side task given to U (used in <code>verbatim</code>)</summary><pre id="side"></pre></details>
  <details class="side" id="random-box" hidden><summary id="random-sum">Side task in the <code>random</code> prompt</summary><pre id="random-side"></pre></details>
  <section>
    <h2>Total transcript loss (nats)</h2>
    <div class="totals" id="totals"></div>
  </section>
  <div class="controls">
    <div class="seg" role="group" aria-label="Compare verbatim against">
      <button type="button" id="cmp-none" aria-pressed="true">verbatim vs none</button>
      <button type="button" id="cmp-random" aria-pressed="false">verbatim vs random</button>
    </div>
    <fieldset class="show">
      <legend>Show</legend>
      <label><input type="checkbox" id="show-text" checked> U's text</label>
      <label><input type="checkbox" id="show-call" checked> U's tool calls</label>
      <label><input type="checkbox" id="show-output" checked> Tool outputs</label>
      <label><input type="checkbox" id="show-marker" checked> Turn markers</label>
    </fieldset>
    <div class="scale"><span>harder</span><div class="bar"></div><span>easier with the true side prompt</span></div>
  </div>
  <p class="note" id="note"></p>
  <section>
    <h2>Gain per step (sum over the step's tokens)</h2>
    <div class="steps" id="steps"></div>
  </section>
  <section>
    <h2>Transcript, shaded per token</h2>
    <div id="transcript" style="display:grid; gap:14px"></div>
  </section>
</div>
<div class="tip" id="tip"></div>

<script>
const D = __DATA__;
const CLIP = 2.0;  // nats; shading saturates here
let base = "none";

const fmt = (x, d = 1) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const delta = t => t[base] - t.verbatim;
function shade(x) {
  const a = Math.min(Math.abs(x) / CLIP, 1) * 0.75;
  if (a < 0.03) return "transparent";
  return `rgba(var(${x > 0 ? "--help" : "--hurt"}),${a.toFixed(3)})`;
}

document.getElementById("meta").innerHTML = [
  ["U", D.u_model], ["environment", D.environment], ["main task", D.main_task], ["side task", D.side_task],
  ["trajectory", D.id], ["T", "Qwen3-32B"]
].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
document.getElementById("side").textContent = D.side_block;
if (D.random_side_block) {
  document.getElementById("random-box").hidden = false;
  document.getElementById("random-sum").innerHTML =
    `Side task in the <code>random</code> prompt: another trajectory's, same environment (${D.random_side_task})`;
  document.getElementById("random-side").textContent = D.random_side_block;
}

const totals = { none: 0, random: 0, verbatim: 0 };
D.tokens.forEach(t => { for (const c in totals) totals[c] += t[c]; });
document.getElementById("totals").innerHTML = ["none", "random", "verbatim"].map(c => `
  <div class="tot"><div class="k">${c}</div><div class="v">${totals[c].toFixed(1)}</div>
  <div class="d">${c === "verbatim" ? `${D.tokens.length} tokens` : `verbatim is ${(totals[c] - totals.verbatim).toFixed(1)} lower`}</div></div>`).join("");

const byMsg = D.messages.map(() => []);
D.tokens.forEach(t => byMsg[t.msg].push(t));
// Which part each token belongs to: U's visible text, U's tool calls, tool outputs, the user turn, or a
// chat-template turn marker (same definition as segment_scores.py: special tokens, the role word and
// newline after <|im_start|>, and the newline after <|im_end|>).
const SPECIAL = new Set(["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]);
const toks = D.tokens;
toks.forEach((t, i) => {
  const p1 = i > 0 ? toks[i - 1].text : "", p2 = i > 1 ? toks[i - 2].text : "";
  t.marker = SPECIAL.has(t.text) || p1 === "<|im_start|>" || (t.text === "\n" && (p2 === "<|im_start|>" || p1 === "<|im_end|>"));
});
byMsg.forEach((ts, k) => {
  const role = D.messages[k].role;
  let inCall = false;
  ts.forEach(t => {
    if (role === "assistant" && t.text === "<tool_call>") inCall = true;
    t.part = t.marker ? "marker" : role === "tool" ? "output" : role === "assistant" ? (inCall ? "call" : "text") : "user";
  });
});
const show = { text: true, call: true, output: true, marker: true, user: true };
const visible = t => show[t.part];

const tr = document.getElementById("transcript");
const nodes = D.messages.map((m, k) => {
  const el = document.createElement("article");
  el.className = "msg " + m.role; el.id = "m" + k;
  el.innerHTML = `<div class="mh"><span class="role">${k}. ${m.role}</span><span class="kind">${m.kind}</span>
    <span class="sum"></span></div><div class="toks"></div>`;
  const toks = el.querySelector(".toks");
  byMsg[k].forEach(t => { const s = document.createElement("span"); s.textContent = t.text; s._t = t; toks.appendChild(s); });
  tr.appendChild(el);
  return el;
});

function render() {
  const sums = byMsg.map(ts => ts.reduce((a, t) => a + (visible(t) ? delta(t) : 0), 0));
  const counts = byMsg.map(ts => ts.filter(visible).length);
  const max = Math.max(...sums.map(Math.abs), 1e-9);
  const total = sums.reduce((a, b) => a + b, 0);
  const shown = counts.reduce((a, b) => a + b, 0);
  const all = shown === D.tokens.length;
  document.getElementById("note").textContent =
    `Each token is shaded by its loss under ${base} minus its loss under verbatim (nats; saturates at ±${CLIP}). ` +
    `Summed over ${all ? "the transcript" : `the ${shown} shown tokens`} the true side prompt saves ${total.toFixed(1)} nats relative to ${base}. Hover a token for its losses.`;
  document.getElementById("steps").innerHTML = D.messages.map((m, k) => {
    if (!counts[k]) return "";
    const v = sums[k], w = Math.abs(v) / max * 50;
    const left = v >= 0 ? 50 : 50 - w;
    return `<button type="button" class="step" data-k="${k}"><span class="i">${k}</span><span>${m.role} · ${m.kind}</span>
      <span class="n">${counts[k]} tok</span>
      <span class="track"><span class="zero"></span><span class="b" style="left:${left}%;width:${w}%;background:${v >= 0 ? "rgb(var(--help))" : "rgb(var(--hurt))"}"></span></span>
      <span class="s">${fmt(v)}</span></button>`;
  }).join("");
  nodes.forEach((el, k) => {
    el.hidden = !counts[k];
    el.querySelector(".sum").textContent = `${fmt(sums[k])} nats`;
    el.querySelectorAll(".toks span").forEach(s => {
      s.classList.toggle("off", !visible(s._t));
      s.style.background = shade(delta(s._t));
    });
  });
}
for (const part of ["text", "call", "output", "marker"]) {
  document.getElementById("show-" + part).addEventListener("change", e => { show[part] = e.target.checked; render(); });
}

document.getElementById("steps").addEventListener("click", e => {
  const b = e.target.closest(".step"); if (!b) return;
  const el = nodes[+b.dataset.k];
  el.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
});
for (const c of ["none", "random"]) {
  document.getElementById("cmp-" + c).addEventListener("click", () => {
    base = c;
    for (const o of ["none", "random"]) document.getElementById("cmp-" + o).setAttribute("aria-pressed", String(o === c));
    render();
  });
}
const tip = document.getElementById("tip");
tr.addEventListener("mousemove", e => {
  const s = e.target.closest(".toks span");
  if (!s) { tip.style.display = "none"; return; }
  const t = s._t;
  tip.textContent = `${JSON.stringify(t.text)}\nnone      ${t.none.toFixed(2)}\nrandom    ${t.random.toFixed(2)}\nverbatim  ${t.verbatim.toFixed(2)}\n${base} − verbatim  ${fmt(delta(t), 2)}`;
  tip.style.display = "block";
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8), y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
});
tr.addEventListener("mouseleave", () => { tip.style.display = "none"; });
render();
</script>
"""

U_MODELS = {"glm53": "GLM-5.3", "deepseek_v4_pro": "DeepSeek V4 Pro"}


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".html")
    d = json.loads(src.read_text())
    d["u_model"] = U_MODELS.get(d["dataset"], d["dataset"])
    for t in d["tokens"]:
        for c in ("none", "random", "verbatim"):
            t[c] = round(t[c], 3)
    data = json.dumps(d, ensure_ascii=False).replace("</", "<\\/")
    title = f"{d['u_model']} {d['side_task']} attribution"
    out.write_text(TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
