# Simeon Nedelchev

Personal academic page: [simeon-ned.github.io](https://simeon-ned.github.io/)

Built with [Eleventy](https://www.11ty.dev/). Edit content in `content/` and templates in `src/`; GitHub Actions deploys `_site/` to Pages.

## Local development

```bash
npm ci
npm run dev       # http://localhost:8080 (live reload; uses polling to avoid Linux ENOSPC)
npm run build     # writes _site/
npm run preview   # build once, serve _site/ (no file watchers)
```

`npm run dev` sets `CHOKIDAR_USEPOLLING=1` so it works when the system inotify budget is exhausted (common with large workspaces / IDEs). For native inotify watches, use `npm run dev:native` after raising the limit:

```bash
sudo sysctl -w fs.inotify.max_user_watches=524288
```

## Content sources

| File | Purpose |
|------|---------|
| [`content/publications.bib`](content/publications.bib) | All papers — homepage + CV lists (optional fields: `url`, `doi`, `eprint`, `project`, `video`, `keywords`) |
| [`content/projects.yaml`](content/projects.yaml) | Homepage project cards + CV selected projects |
| [`src/`](src/) | Nunjucks templates, shared nav/footer, CSS |

Adding a paper: append a BibTeX entry to `publications.bib`, then rebuild (or push). Use `keywords = {preprint}`, `{journal}`, or `{conference}` for CV grouping.

## Presentations (Reveal.js)

Job-application slide decks live in `presentations/`. Copy `presentations/template/` for each role — see [presentations/README.md](presentations/README.md). Decks are hand-authored (desktop talk first) and copied into `_site` as-is.

## CV (LaTeX)

Source: `cv/cv.tex` — English PDF for the site. The Publications section is generated from [`content/publications.bib`](content/publications.bib) into `cv/publications.tex`:

```bash
npm run cv          # regenerate pubs from bib + latexmk → pdf/CV.pdf
# or:
npm run cv:pubs && cd cv && make
```

Requires Node (for the bib → TeX step) and `latexmk` (TeX Live). Output: [`pdf/CV.pdf`](pdf/CV.pdf). Russian CV stays local (`make ru`). CI regenerates pubs and rebuilds the English PDF on every deploy.
