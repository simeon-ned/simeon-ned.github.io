const fs = require("fs");
const path = require("path");
const { Cite } = require("@citation-js/core");
require("@citation-js/plugin-bibtex");

const BIB_PATH = path.join(__dirname, "../../content/publications.bib");

/** Citation.js drops unknown BibTeX fields; scrape them from the source. */
function parseBibExtras(bibText) {
  const extras = {};
  const entryRe = /@\w+\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\n\}/g;
  let m;
  while ((m = entryRe.exec(bibText))) {
    const id = m[1];
    const body = m[2];
    const fields = {};
    const fieldRe = /(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g;
    let f;
    while ((f = fieldRe.exec(body))) {
      fields[f[1].toLowerCase()] = f[2].trim();
    }
    extras[id] = fields;
  }
  return extras;
}

function yearOf(entry) {
  const y = entry.issued?.["date-parts"]?.[0]?.[0];
  return y != null ? String(y) : "n.d.";
}

function formatAuthorPlain(author) {
  if (!author) return "";
  const family = author.family || "";
  const given = author.given || "";
  if (family && given) return `${given} ${family}`;
  return family || given || author.literal || "";
}

function formatAuthorsHtml(authors) {
  if (!authors?.length) return "";
  return authors
    .map((a) => {
      const name = formatAuthorPlain(a);
      const isMe =
        /nedelchev/i.test(a.family || "") || /nedelchev/i.test(a.literal || "");
      return isMe ? `<strong>${escapeHtml(name)}</strong>` : escapeHtml(name);
    })
    .join(", ");
}

function authorShort(author) {
  const family = author.family || "";
  const given = author.given || "";
  if (!given) return family || author.literal || "";
  const initials = given
    .split(/[\s-]+/)
    .filter(Boolean)
    .map((p) => p[0].toUpperCase())
    .join("");
  return `${family} ${initials}`;
}

function isMeAuthor(author) {
  return (
    /nedelchev/i.test(author.family || "") ||
    /nedelchev/i.test(author.literal || "")
  );
}

function formatAuthorsCv(authors) {
  if (!authors?.length) return "";
  return authors
    .map((a) => {
      const short = authorShort(a);
      return isMeAuthor(a)
        ? `<strong>${escapeHtml(short)}</strong>`
        : escapeHtml(short);
    })
    .join(", ");
}

function formatAuthorsTex(authors) {
  if (!authors?.length) return "";
  return authors
    .map((a) => {
      const short = escapeTex(authorShort(a));
      return isMeAuthor(a) ? `\\textbf{${short}}` : short;
    })
    .join(", ");
}

function escapeTex(s) {
  return String(s)
    .replace(/\\/g, "\\textbackslash{}")
    .replace(/([{}$&#_%])/g, "\\$1")
    .replace(/~/g, "\\textasciitilde{}")
    .replace(/\^/g, "\\textasciicircum{}");
}

function formatPages(page) {
  if (!page) return "";
  return String(page).replace(/(\d+)-(\d+)/g, "$1--$2");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function keywordList(entry, extras) {
  const raw =
    extras.keywords ||
    entry.keyword ||
    entry.keywords ||
    entry.custom?.keyword ||
    "";
  if (!raw) return [];
  const list = Array.isArray(raw) ? raw : String(raw).split(/[,;]/);
  return list.map((x) => String(x).trim().toLowerCase()).filter(Boolean);
}

function hasKeyword(entry, extras, kw) {
  return keywordList(entry, extras).includes(kw.toLowerCase());
}

function venueOf(entry, extras, cat) {
  if (entry["container-title"]) return entry["container-title"];
  if (entry["collection-title"]) return entry["collection-title"];
  if (entry.event) return typeof entry.event === "string" ? entry.event : entry.event.name;
  if (cat === "preprint" || hasKeyword(entry, extras, "preprint")) {
    return "arXiv preprint";
  }
  return "";
}

function categoryOf(entry, extras) {
  const url = entry.URL || extras.url || "";
  const doi = entry.DOI || extras.doi || "";
  if (
    hasKeyword(entry, extras, "preprint") ||
    url.includes("arxiv.org") ||
    String(doi).startsWith("10.48550")
  ) {
    return "preprint";
  }
  if (hasKeyword(entry, extras, "journal") || entry.type === "article-journal") {
    if (hasKeyword(entry, extras, "conference")) return "conference";
    return "journal";
  }
  if (
    hasKeyword(entry, extras, "conference") ||
    entry.type === "paper-conference" ||
    entry.type === "chapter"
  ) {
    return "conference";
  }
  if (entry["container-title"] || entry["collection-title"]) return "conference";
  return "other";
}

function arxivId(entry, extras) {
  const fromUrl = (entry.URL || extras.url || "").match(
    /arxiv\.org\/abs\/([\w.]+)/
  )?.[1];
  return fromUrl || extras.eprint || entry.custom?.eprint || null;
}

function linksOf(entry, extras) {
  const links = [];
  const project = extras.project || entry.custom?.project || entry.project;
  if (project) links.push({ label: "Project page", href: project });

  let paperUrl = entry.URL || extras.url;
  if (!paperUrl && (entry.DOI || extras.doi)) {
    paperUrl = `https://doi.org/${entry.DOI || extras.doi}`;
  }
  const eprint = arxivId(entry, extras);
  if (!paperUrl && eprint) paperUrl = `https://arxiv.org/abs/${eprint}`;
  if (paperUrl) links.push({ label: "Paper", href: paperUrl });

  const video = extras.video || entry.custom?.video;
  if (video) links.push({ label: "Video", href: video });

  return links;
}

function normalizeEntry(raw, extrasById) {
  const extras = extrasById[raw.id] || extrasById[raw["citation-key"]] || {};
  const year = yearOf(raw);
  const cat = categoryOf(raw, extras);
  const venue = venueOf(raw, extras, cat) || (cat === "preprint" ? "arXiv preprint" : "");

  const volume = raw.volume || extras.volume || "";
  const issue = raw.issue || extras.number || "";
  const pages = formatPages(raw.page || extras.pages || "");

  return {
    id: raw.id,
    title: raw.title || "Untitled",
    year,
    yearNum: parseInt(year, 10) || 0,
    category: cat,
    venue,
    journal: raw["container-title"] || "",
    volume,
    issue,
    pages,
    authorsHtml: formatAuthorsHtml(raw.author),
    authorsCv: formatAuthorsCv(raw.author),
    authorsTex: formatAuthorsTex(raw.author),
    authorsPlain: (raw.author || []).map(formatAuthorPlain).join(", "),
    links: linksOf(raw, extras),
    arxiv: arxivId(raw, extras),
    doi: raw.DOI || extras.doi || null,
    url: raw.URL || extras.url || null,
  };
}

function loadPublications() {
  const bib = fs.readFileSync(BIB_PATH, "utf8");
  const extrasById = parseBibExtras(bib);
  const cite = new Cite(bib, { forceType: "@bibtex/text" });
  const items = cite.data.map((raw) => normalizeEntry(raw, extrasById));

  items.sort((a, b) => b.yearNum - a.yearNum || a.title.localeCompare(b.title));

  const byYearMap = new Map();
  for (const item of items) {
    if (!byYearMap.has(item.year)) byYearMap.set(item.year, []);
    byYearMap.get(item.year).push(item);
  }
  const byYear = [...byYearMap.entries()]
    .sort((a, b) => parseInt(b[0], 10) - parseInt(a[0], 10))
    .map(([year, pubs]) => ({
      year,
      pubs,
      open: (parseInt(year, 10) || 0) >= 2025,
    }));

  const byType = {
    preprint: items.filter((p) => p.category === "preprint"),
    journal: items.filter((p) => p.category === "journal"),
    conference: items.filter((p) => p.category === "conference"),
    other: items.filter((p) => p.category === "other"),
  };

  return { items, byYear, byType };
}

module.exports = loadPublications;
module.exports.loadPublications = loadPublications;
module.exports.escapeTex = escapeTex;
