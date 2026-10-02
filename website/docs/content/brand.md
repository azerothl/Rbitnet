# Rbitnet — brand guidelines

**Project:** Rbitnet (*Rabbit* + *bit*) · pure-Rust native-first GGUF engine  
**Visual direction:** *Phosphor Warren* — white pixel rabbit, bit grid, mint phosphor

Default language for the site and product docs is **English**. French is secondary (e.g. `DEMARRAGE_5MIN.md`).

This guide covers logo, color, type, and tone for the site, README, releases, and integrations (Akasha, docs).

## Concept

Rbitnet is a **digital warren**: a white pixel-art rabbit carrying bits (mint / amber). The *Rabbit* + *bit* wordplay is the brand signal — not decoration.

| Signal | Rule |
|--------|------|
| Brand first | The name **Rbitnet** owns the first viewport (not only the nav) |
| Pixel = truth | The mark is a crisp grid; no soft glow, no plastic 3D |
| One motif | Bits / grid / phosphor — no generic cloud/AI icon kits |

## Logo

Files live in [`website/assets/brand/`](../website/assets/brand/).

| File | Use |
|------|-----|
| `rbitnet-mark.svg` / `.png` | Mark alone (transparent) |
| `rbitnet-mark-on-ink.svg` | Avatar / favicon / dark surfaces |
| `rbitnet-mark-on-mist.svg` | Light surfaces / docs |
| `rbitnet-lockup.svg` | Mark + wordmark |
| `rbitnet-wordmark.svg` | Wordmark only (**Rbit** ink + **net** mint) |
| `rbitnet-icon-512.png` | App / OG / Discord icon |
| `favicon.svg` | Site favicon |
| `rbitnet-logo-ai-reference.png` | Illustrative reference (not the vector source of truth) |
| `rbitnet-hero-atmosphere.png` | Hero atmosphere art |

### Clear space

Keep at least **2 logical pixels** (≈ 1/8 of mark width) around the rabbit. Do not place type in the ears. Do not stretch, soft-shadow, or recolor outside the palette.

### Versions

- **Light:** transparent mark or `on-mist`
- **Dark:** `on-ink` or white mark
- **Mono:** ink on mist, or white on ink

## Color

```css
:root {
  --ink: #0B1220;       /* text, pixel outlines */
  --mist: #E8EEF4;      /* primary background */
  --mist-deep: #D5DEE8; /* layers / grid */
  --paper: #F4F7FA;     /* light surfaces */
  --mint: #00C896;      /* phosphor — brand accent */
  --amber: #F0A202;     /* bits / primary CTA */
  --fog: #9BB0C7;       /* secondary lines */
  --white: #F7FAFC;     /* pixel fur */
}
```

| Token | Hex | Role |
|-------|-----|------|
| Ink | `#0B1220` | Text, outlines |
| Mist | `#E8EEF4` | Atmosphere background |
| Mint | `#00C896` | Primary accent, “net”, secondary CTAs |
| Amber | `#F0A202` | Bits, highlights, primary CTAs |
| Fog | `#9BB0C7` | Rules, meta, muted |

**Avoid:** purple / indigo gradients, cream + terracotta + display-serif “AI default”, multi-layer neon cyberpunk, stacked pill chips.

## Typography

| Role | Family | Use |
|------|--------|-----|
| Display / brand | [Syne](https://fonts.google.com/specimen/Syne) 700–800 | Titles, hero **Rbitnet** |
| Body | [Sora](https://fonts.google.com/specimen/Sora) 400–600 | Paragraphs, UI |
| Mono / bits | [IBM Plex Mono](https://fonts.google.com/specimen/IBM+Plex+Mono) | Optional wordmark, code, CLI |

Scale: brand hero ~`clamp(3.5rem, 12vw, 7rem)`; H2 ~2rem; body 1.05–1.125rem; line-height 1.5–1.65.

## Motion

Intentional motion only:

1. **Bit-grid drift** — very slow background grid
2. **Hero reveal** — brand + lede fade/slide (≤ 700 ms)
3. **Bit pulse** — small mint/amber marker on the brand title

Respect `prefers-reduced-motion: reduce`.

## Editorial tone

- Direct, technical, no empty marketing
- **English by default** on the site and core docs; French secondary
- Concrete verbs: *mmap*, *GGUF*, *tok/s*, *OpenAI-compatible*
- Avoid: “revolutionary”, “next-gen AI”, decorative emoji

## Presentation site

Source: [`website/`](../website/) (landing + on-site docs).  
Workflow: [`.github/workflows/pages.yml`](../.github/workflows/pages.yml).

## Checklist

- [ ] First viewport without nav still reads as **Rbitnet**
- [ ] Palette limited to the tokens above
- [ ] Pixel mark stays crisp (SVG or integer multiples of the grid)
- [ ] One job per section
- [ ] CTAs: Install / Docs / GitHub
