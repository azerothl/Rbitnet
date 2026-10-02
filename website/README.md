# Rbitnet website

Static presentation site + on-site docs for **Rbitnet** (GitHub Pages).

**Default language: English.** French is secondary (`docs/page.html?doc=demarrage-5min`).

## Local preview

```bash
# Sync curated markdown from repo docs/ into website/docs/content/
./website/scripts/sync-docs.sh

cd website
python3 -m http.server 4173
# http://127.0.0.1:4173/
# http://127.0.0.1:4173/docs/
```

## Structure

| Path | Role |
|------|------|
| `index.html` | Landing (status, roadmap, install) |
| `docs/` | Docs hub + markdown viewer |
| `docs/content/` | Synced English guides (plus FR quickstart) |
| `assets/brand/` | Logo and hero art |
| `scripts/sync-docs.sh` | Copy curated docs from `docs/` |

## Deploy

On push to `main`, [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) runs `sync-docs.sh` and publishes `website/`.

Enable: **Settings → Pages → Source: GitHub Actions**.

Brand guidelines: [`docs/BRAND.md`](../docs/BRAND.md).
