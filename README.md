# Simeon Nedelchev

Personal academic page: [simeon-ned.github.io](https://simeon-ned.github.io/)

Built with [Eleventy](https://www.11ty.dev/). Edit content in `content/` and templates in `src/`. Pushing to `master` deploys `_site/` to GitHub Pages via Actions (no TeX in CI).

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
| [`content/publications.bib`](content/publications.bib) | All papers — homepage + HTML CV + (via local build) PDF CV |
| [`content/projects.yaml`](content/projects.yaml) | Homepage project cards + CV selected projects |
| [`src/`](src/) | Nunjucks templates, shared nav/footer, CSS |
| [`pdf/CV.pdf`](pdf/CV.pdf) | English CV PDF (built locally, committed, copied into `_site`) |

Adding a paper: append a BibTeX entry to `publications.bib` (use `keywords = {preprint}`, `{journal}`, or `{conference}`). Site HTML updates on the next Eleventy build; regenerate the PDF before push if you want the PDF list updated too.

## CV (LaTeX) — build locally

Source: `cv/cv.tex`. Publications are generated from `content/publications.bib` into `cv/publications.tex`, then compiled to [`pdf/CV.pdf`](pdf/CV.pdf).

```bash
npm run cv          # bib → publications.tex + latexmk → pdf/CV.pdf
# or:
npm run cv:pubs && cd cv && make
```

Requires Node and `latexmk` (TeX Live). Russian CV stays local (`cd cv && make ru`). Commit the updated `pdf/CV.pdf` when it changes.

## Deploy (push to GitHub Pages)

CI only runs `npm ci` + `npm run build` and uploads `_site/`. Rebuild the CV PDF on your machine first if bib/CV sources changed.

```bash
# 1) If you edited publications.bib or cv/cv.tex:
npm run cv

# 2) Preview (optional)
npm run build && npm run preview

# 3) Commit and push to master
git add -A
git status   # confirm pdf/CV.pdf is included if you rebuilt it
git commit -m "Describe your change."
git push origin master
```

Watch the deploy: repo → **Actions** → “Deploy site”, or `gh run watch`.  
Site: https://simeon-ned.github.io/

Manual re-deploy without new commits: **Actions → Deploy site → Run workflow**.

## Presentations (Reveal.js)

Job-application slide decks live in `presentations/`. Copy `presentations/template/` for each role — see [presentations/README.md](presentations/README.md). Decks are hand-authored (desktop talk first) and copied into `_site` as-is.
