# Job application presentations (Reveal.js)

Static HTML slides derived from [simeon-ned.github.io](https://simeon-ned.github.io/) content — bio, recent projects, experience, and publications.

## Quick start

```bash
cd presentations/template
python3 -m http.server 8080
# Open http://localhost:8080
```

Use arrow keys / space to navigate. Press `S` for speaker notes (where added). Press `F` for fullscreen.

## Create a dedicated presentation for a job

1. **Copy the template folder** and rename it for the role or company:

   ```bash
   cp -r template acme-robotics-senior-rl
   cd acme-robotics-senior-rl
   ```

2. **Customize the About Me slide** in `index.html`:
   - Change `<title>` if needed.
   - Optionally adjust `.about-headline` to emphasize skills the posting cares about.

3. **Reorder or hide project slides** — each project is one horizontal slide (media + description). Delete entire `<section>` blocks you do not want, or move the most relevant ones right after About Me.

4. **Add speaker notes** — optional `<aside class="notes">` inside any project `<section>`. Notes appear in the speaker view (`S` key) but not on the projected slide.

5. **Tailor the closing slide** — update the thank-you message for the specific team.

6. **Export to PDF** (optional):

   ```bash
   # Serve locally, then open in Chrome:
   # http://localhost:8080/?print-pdf
   # Print → Save as PDF (disable headers/footers, set margins to none)
   ```

## Structure

| Slide block | Purpose |
|-------------|---------|
| About Me | Photo, name, headline, contact (+ Google Scholar), experience timeline, pitch |
| Experience & Technical Profile | Career path + what I deliver + RL stack (one slide for hiring teams) |
| Per-project | One slide each: GIFs/video + short description + links |
| Open Source | Pinocchio, Pink, Mink, mink-warp, GMR, mujoco-sysid, MJINX |
| Motivation & Fit | Why apply, future directions, expertise |
| Thank you | Contact + Q&A |

## Assets

Images and video are referenced from the site root (`../../images/...`). They are the same files used on the homepage project cards. If you host a presentation outside this repo, copy `images/profile.jpg` and `images/projects/` into the presentation folder and update paths.

## Dependencies

Reveal.js is loaded from jsDelivr CDN (no npm install). Offline use: download [Reveal.js 5.x](https://github.com/hakimel/reveal.js/releases) into `vendor/reveal.js/` and update the `<link>` / `<script>` paths in `index.html`.

## Tips for interviews

- Keep **one main narrative**: sim-to-real humanoid control, or IK libraries, or RL locomotion — pick based on the job description.
- Use **one slide per project** in the live talk; extra detail goes in speaker notes.
- For a **15-minute slot**, show: About Me (30s) → Experience & Technical Profile (1 min) → 2–3 projects → Thank you.
- For a **research talk**, expand PSM / LocoGen / WBC-Mjlab and shorten the profile slide.
