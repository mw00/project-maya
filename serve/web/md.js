// serve/web/md.js - an answer's Markdown as HTML (escaped first, then formatted), its code blocks with a small local
// syntax highlighter (no library), and the code blocks' buttons.  Streaming answers are patched in place (patchHtml).
import {SPRITE, esc, download} from "./lib.js";
import {copyText, openDialog} from "./state.js";

const ico = (name) => `<svg class="st-icon st-icon--sm" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;

// ------------------------------------------------------------------ syntax highlighting
const KW = (s) => new Set(s.split(/\s+/));
const HL_FAMILY = {};
for (const [fam, names] of Object.entries({
  c: "js javascript jsx mjs cjs ts typescript tsx java c h cpp cc cxx hpp c++ cu cuh cuda cs csharp go golang rust rs kotlin kt swift php dart scala glsl hlsl wgsl zig",
  py: "py python python3", sh: "sh bash shell zsh console ps1 powershell bat cmd", json: "json jsonc json5",
  html: "html htm xhtml xml svg vue", css: "css scss less", sql: "sql", yaml: "yaml yml toml ini cfg conf env dockerfile",
})) for (const n of names.split(" ")) HL_FAMILY[n] = fam;
const STR = /"(?:\\[\s\S]|[^"\\\n])*"?|'(?:\\[\s\S]|[^'\\\n])*'?/y;
const NUM = /\b(?:0x[\da-fA-F_]+|0b[01_]+|\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?)[a-zA-Z]*\b/y;
const WORD = /[A-Za-z_$][\w$]*/y;
const HL = {
  c: {rules: [[/\/\/[^\n]*|\/\*[\s\S]*?(?:\*\/|$)/y, "c"], [/`(?:\\[\s\S]|[^`\\])*`?/y, "s"], [STR, "s"],
              [/#[ \t]*[a-z]+/y, "k"], [NUM, "n"], [WORD, "w"]],
      kw: KW(`abstract as async await break case catch class const constexpr continue debugger default defer delete do else enum
        export extends extern false final finally fn for from func function get go goto if impl implements import in inline
        instanceof interface let loop match mod module move mut namespace new nil null nullptr of operator override package
        private protected pub public readonly ref register return self Self sizeof static struct super switch template this
        throw throws trait true try type typedef typename typeof union unsafe use using var virtual void volatile where while
        with yield auto bool char double float int long short signed unsigned void string number boolean any never unknown
        __global__ __device__ __host__ __shared__ __restrict__ i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 usize isize`)},
  py: {rules: [[/#[^\n]*/y, "c"], [/[rbfuRBFU]{0,2}(?:"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$))/y, "s"],
               [/[rbfuRBFU]{0,2}(?:"(?:\\[\s\S]|[^"\\\n])*"?|'(?:\\[\s\S]|[^'\\\n])*'?)/y, "s"], [/@[\w.]+/y, "f"], [NUM, "n"], [WORD, "w"]],
       kw: KW(`and as assert async await break case class continue def del elif else except False finally for from global if
         import in is lambda match None nonlocal not or pass raise return True try while with yield self cls print len range`)},
  sh: {rules: [[/(?:^|(?<=\s))#[^\n]*/y, "c"], [STR, "s"], [/\$\{[^}\n]*\}?|\$[\w@#?*!$-]+/y, "v"], [NUM, "n"], [/[A-Za-z_][\w.-]*/y, "w"]],
       kw: KW(`if then else elif fi for while until do done case esac in function return export local set unset echo printf cd
         sudo source alias exit read shift trap eval exec`)},
  json: {rules: [[/"(?:\\[\s\S]|[^"\\\n])*"(?=\s*:)/y, "a"], [STR, "s"], [/-?\d+\.?\d*(?:[eE][+-]?\d+)?/y, "n"], [/[a-z]+/y, "w"]],
         kw: KW("true false null")},
  html: {rules: [[/<!--[\s\S]*?(?:-->|$)/y, "c"], [/<\/?[A-Za-z][\w:-]*/y, "k"], [/[A-Za-z_:][\w:.-]*(?==)/y, "a"], [STR, "s"],
                 [/&#?\w+;/y, "n"]], kw: KW("")},
  css: {rules: [[/\/\*[\s\S]*?(?:\*\/|$)/y, "c"], [STR, "s"], [/#[\da-fA-F]{3,8}\b/y, "n"], [/@[\w-]+/y, "k"],
                [/-?\d+\.?\d*(?:px|em|rem|%|vh|vw|s|ms|deg|fr)?/y, "n"], [/[\w-]+(?=\s*:[^:{;]*[;}])/y, "a"]], kw: KW("")},
  sql: {rules: [[/--[^\n]*|\/\*[\s\S]*?(?:\*\/|$)/y, "c"], [STR, "s"], [NUM, "n"], [WORD, "w"]],
        kw: KW(`select from where and or not insert into values update set delete create table index view drop alter add join
          left right inner outer on group by order having limit offset as distinct union all null is in like between case
          when then else end primary key foreign references default exists with returning asc desc count sum avg min max`),
        ci: true},
  yaml: {rules: [[/#[^\n]*/y, "c"], [/[\w.-]+(?=[ \t]*[:=])/y, "a"], [STR, "s"], [NUM, "n"], [/[a-z]+/y, "w"]],
         kw: KW("true false null yes no on off")},
};
export function highlight(code, lang) {
  const fam = HL_FAMILY[(lang || "").toLowerCase()];
  if (!fam || code.length > 200000) return esc(code);
  const R = HL[fam];
  let out = "", plain = "", i = 0;
  while (i < code.length) {
    let tok = null, cls = null;
    for (const [re, c] of R.rules) {
      re.lastIndex = i;
      const m = re.exec(code);
      if (m && m[0].length) { tok = m[0]; cls = c; break; }
    }
    if (tok === null) { plain += code[i++]; continue; }
    if (cls === "w") {
      const w = R.ci ? tok.toLowerCase() : tok;
      cls = R.kw.has(w) ? "k" : code[i + tok.length] === "(" ? "f" : null;
    }
    if (plain) { out += esc(plain); plain = ""; }
    out += cls ? `<span class="tk-${cls}">${esc(tok)}</span>` : esc(tok);
    i += tok.length;
  }
  return out + esc(plain);
}

// ------------------------------------------------------------------ Markdown
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
export const CODE_EXT = {python: "py", py: "py", javascript: "js", js: "js", jsx: "jsx", typescript: "ts", ts: "ts", tsx: "tsx",
  html: "html", htm: "html", css: "css", scss: "scss", json: "json", bash: "sh", sh: "sh", shell: "sh", zsh: "sh",
  c: "c", h: "h", cpp: "cpp", "c++": "cpp", hpp: "hpp", cuda: "cu", cu: "cu", rust: "rs", rs: "rs", go: "go", java: "java",
  kotlin: "kt", swift: "swift", sql: "sql", yaml: "yaml", yml: "yml", toml: "toml", xml: "xml", svg: "svg", markdown: "md",
  md: "md", ps1: "ps1", powershell: "ps1", php: "php", ruby: "rb", rb: "rb", lua: "lua", dockerfile: "Dockerfile"};
export const isHtmlCode = (lang, code) => ["html", "htm", "xhtml"].includes((lang || "").toLowerCase()) ||
                                          (!lang && /^\s*(<!doctype html|<html[\s>])/i.test(code));
// a code block: its language, Preview (a complete HTML page), Wrap, Download and Copy
export function codeBlock(lang, code, complete = true) {
  const l = (lang || "").trim().split(/\s+/)[0] || "";
  const btn = (attr, name, title) =>
    `<button type="button" class="code__btn" ${attr} aria-label="${title}" title="${title}">${ico(name)}</button>`;
  const actions = (complete && isHtmlCode(l, code) ? btn("data-code-preview", "eye", "Preview this page") : "") +
    btn("data-code-wrap", "wrap", "Wrap long lines") + btn("data-code-download", "download", "Download as a file") +
    `<button type="button" class="code__btn code__btn--label" data-code-copy aria-label="Copy the code">${ico("copy")}<span>Copy</span></button>`;
  return `<div class="code" data-lang="${esc(l)}"><div class="code__head"><span class="code__lang">${esc(l || "code")}</span>` +
    `<span class="code__actions">${actions}</span></div><pre><code>${highlight(code, l)}</code></pre></div>`;
}
// a list, nested by indentation: [{indent, ordered, start, text}]
function renderList(items) {
  let pos = 0;
  const build = (indent) => {
    const first = items[pos];
    const tag = first.ordered ? "ol" : "ul";
    let html = `<${tag}${first.ordered && first.start && first.start !== 1 ? ` start="${first.start}"` : ""}>`;
    while (pos < items.length && items[pos].indent >= indent) {
      const it = items[pos];
      if (it.indent > indent) {                     // a deeper level: inside the item before it
        const sub = build(it.indent);
        html = html.endsWith("</li>") ? `${html.slice(0, -5)}${sub}</li>` : `${html}<li>${sub}</li>`;
        continue;
      }
      html += `<li>${inline(it.text)}</li>`;
      pos++;
    }
    return `${html}</${tag}>`;
  };
  let out = "";
  while (pos < items.length) out += build(items[pos].indent);
  return out;
}
const LIST_RE = /^(\s*)(?:([-*+])|(\d+)[.)])\s+(.*)$/;
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [];
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); const t = m[1].length <= 2 ? "h3" : "h4"; out.push(`<${t}>${inline(m[2])}</${t}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<div class="table-scroll"><table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table></div>");
      continue;
    }
    if (LIST_RE.test(l)) {
      flushPara();
      const items = [];
      while (i < lines.length) {
        const L = lines[i], mm = L.match(LIST_RE);
        if (mm) { items.push({indent: mm[1].replace(/\t/g, "    ").length, ordered: !mm[2], start: mm[3] ? +mm[3] : null, text: mm[4]}); i++; continue; }
        if (L.trim() && /^\s{2,}\S/.test(L) && items.length) { items[items.length - 1].text += " " + L.trim(); i++; continue; }
        break;
      }
      i--;
      out.push(renderList(items));
      continue;
    }
    para.push(l);
  }
  flushPara();
  return out.join("");
}
export function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest, false); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}
// While a long answer streams, the part before its last blank line (outside a code block) is rendered once and kept;
// only the tail is rendered again each frame (a 14K-token answer stays smooth).
export function streamingMarkdown(m) {
  const t = m.text;
  if (t.length < 6000) return markdown(t);
  const c = m._md || (Object.defineProperty(m, "_md", {value: {upto: 0, html: ""}, enumerable: false, writable: true}), m._md);
  let cut = t.lastIndexOf("\n\n", t.length - 3);
  while (cut > c.upto && ((t.slice(0, cut).match(/(^|\n)```/g) || []).length % 2)) cut = t.lastIndexOf("\n\n", cut - 1);
  if (cut > c.upto) { c.html += markdown(t.slice(c.upto, cut)); c.upto = cut; }
  return c.html + markdown(t.slice(c.upto));
}

// an answer's new HTML into its element with the fewest DOM changes (it repaints up to every frame while it streams):
// a block that is the same stays as it is - no flash, a selection and a code block's wrap kept - and the code block still
// being written keeps its frame, buttons and scroll box, its code alone replaced
export function patchHtml(el, html) {
  const tpl = document.createElement("template");
  tpl.innerHTML = html;
  const next = [...tpl.content.childNodes], cur = [...el.childNodes];
  next.forEach((b, i) => {
    const a = cur[i];
    if (!a) el.appendChild(b);
    else if (a.nodeType === 3 && b.nodeType === 3) { if (a.data !== b.data) a.data = b.data; }
    else if (a.nodeType === 1 && b.nodeType === 1 && a.isEqualNode(b)) { /* the same */ }
    else if (a.nodeType === 1 && b.nodeType === 1 && a.matches(".code") && b.matches(".code") &&
             a.dataset.lang === b.dataset.lang) patchCode(a, b);
    else el.replaceChild(b, a);
  });
  for (let i = cur.length - 1; i >= next.length; i--) cur[i].remove();
}
// a code block that grew: the new code into the block, which stays scrolled to its end while it is written (unless its
// reader scrolled it up); the buttons swapped only when their set changes (Preview, once the block is complete)
function patchCode(a, b) {
  const pre = a.querySelector("pre"), code = a.querySelector("code"), nc = b.querySelector("code");
  const atEnd = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
  if (code.innerHTML !== nc.innerHTML) code.innerHTML = nc.innerHTML;
  if (atEnd) pre.scrollTop = pre.scrollHeight;
  const ah = a.querySelector(".code__actions"), bh = b.querySelector(".code__actions");
  if (ah && bh && ah.children.length !== bh.children.length) ah.replaceWith(bh);
}

// ------------------------------------------------------------------ the code blocks' buttons (answers and Settings' examples)
export function onCodeClick(e) {
  const b = e.target.closest("[data-code-copy], [data-code-wrap], [data-code-download], [data-code-preview]");
  if (!b) return;
  const block = b.closest(".code"), code = block.querySelector("pre").textContent, lang = block.dataset.lang || "";
  if (b.hasAttribute("data-code-copy")) {
    copyText(code, true);
    const use = b.querySelector("use"), label = b.querySelector("span");
    use.setAttribute("href", `${SPRITE}#i-check`);
    if (label) label.textContent = "Copied";
    setTimeout(() => { use.setAttribute("href", `${SPRITE}#i-copy`); if (label) label.textContent = "Copy"; }, 1500);
  } else if (b.hasAttribute("data-code-wrap")) {
    block.classList.toggle("is-wrapped");
    b.setAttribute("aria-pressed", String(block.classList.contains("is-wrapped")));
  } else if (b.hasAttribute("data-code-download")) {
    const ext = CODE_EXT[lang.toLowerCase()] || (isHtmlCode(lang, code) ? "html" : "txt");
    download(ext === "Dockerfile" ? ext : `maya-code.${ext}`, code);
  } else if (b.hasAttribute("data-code-preview")) previewHtml(code);
}
// an HTML answer in a sandboxed frame: scripts run, but cut off from this page (no same-origin, no storage)
export function previewHtml(code) {
  const wrap = document.createElement("div");
  wrap.className = "preview";
  wrap.innerHTML = `<p class="muted small">Runs in a sandbox, cut off from this page. Scripts it loads from the web (a
    three.js CDN, say) come from the internet.</p>`;
  const f = document.createElement("iframe");
  f.className = "preview__frame";
  f.setAttribute("sandbox", "allow-scripts allow-pointer-lock allow-modals");
  f.setAttribute("title", "Preview");
  f.srcdoc = code;
  wrap.appendChild(f);
  openDialog({title: "Preview", node: wrap, wide: true});
}
