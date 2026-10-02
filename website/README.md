# Rbitnet website

Static presentation site for **Rbitnet** (GitHub Pages).

## Local preview

```bash
cd website
python3 -m http.server 4173
# open http://127.0.0.1:4173
```

## Deploy

On push to `main`, [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) publishes this folder to GitHub Pages.

Enable in the repo: **Settings → Pages → Source: GitHub Actions**.

Brand guidelines: [`docs/BRAND.md`](../docs/BRAND.md).
