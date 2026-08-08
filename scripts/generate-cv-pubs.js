#!/usr/bin/env node
/**
 * Build cv/publications.tex from content/publications.bib (same parser as the site).
 */
const fs = require("fs");
const path = require("path");
const { loadPublications, escapeTex } = require("../src/_data/publications.js");

const OUT = path.join(__dirname, "../cv/publications.tex");

function venueBits(pub) {
  const parts = [];
  if (pub.volume) {
    let vol = pub.volume;
    if (pub.issue) vol += `(${pub.issue})`;
    if (pub.pages) vol += `:${pub.pages}`;
    parts.push(vol);
  } else if (pub.pages) {
    parts.push(pub.pages);
  }
  if (pub.year) parts.push(pub.year);
  return parts.join(", ");
}

function formatLine(pub) {
  const authors = pub.authorsTex;
  const title = escapeTex(pub.title);

  if (pub.category === "preprint" && pub.arxiv) {
    return `${authors}. ${title}. arXiv:${escapeTex(pub.arxiv)}, ${pub.year}.`;
  }

  const journal = escapeTex(pub.journal || pub.venue || "");
  if (journal) {
    const rest = venueBits(pub);
    return `${authors}. ${title}. \\emph{${journal}}${rest ? `, ${rest}` : ""}.`;
  }

  return `${authors}. ${title}. ${pub.year}.`;
}

function section(label, items) {
  if (!items.length) return "";
  const lines = items.map((pub) => `  \\cvitem{}{${formatLine(pub)}}`).join("\n");
  return [
    `\\textbf{${label}}`,
    "\\begin{itemize}[leftmargin=*, nosep, topsep=0.25em]",
    lines,
    "\\end{itemize}",
    "",
  ].join("\n");
}

const { byType } = loadPublications();
const body = [
  "% Auto-generated from content/publications.bib — do not edit by hand.",
  "% Regenerate: node scripts/generate-cv-pubs.js  (or: cd cv && make)",
  "",
  section("Preprints", byType.preprint),
  section("Journals", byType.journal),
  section("Conference papers", byType.conference),
  section("Other", byType.other),
].join("\n");

fs.writeFileSync(OUT, body.trimEnd() + "\n");
console.log(`Wrote ${path.relative(process.cwd(), OUT)} (${byType.preprint.length + byType.journal.length + byType.conference.length + byType.other.length} entries)`);
