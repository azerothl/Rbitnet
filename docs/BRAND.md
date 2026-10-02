# Rbitnet — charte graphique

**Projet :** Rbitnet (*Rabbit* + *bit*) · moteur GGUF Rust native-first  
**Direction visuelle :** *Phosphor Warren* — lapin pixel blanc, grille de bits, phosphor mint

Cette charte fixe le logo, les couleurs, la typo et le ton pour le site, le README, les releases et les intégrations (Akasha, docs).

## Concept

Rbitnet est un **terrier numérique** : un lapin blanc en pixel art qui porte des bits (mint / amber). Le jeu de mot *Rabbit + bit* est le signal de marque — pas un ornement.

| Signal | Règle |
|--------|--------|
| Marque d’abord | Le nom **Rbitnet** domine le premier viewport (pas seulement en nav) |
| Pixel = vérité | Le mark est une grille nette ; pas de soft glow ni de 3D plastique |
| Un motif | Bits / grille / phosphor — pas d’icônes génériques cloud/AI |

## Logo

Fichiers dans [`website/assets/brand/`](../website/assets/brand/).

| Fichier | Usage |
|---------|--------|
| `rbitnet-mark.svg` / `.png` | Mark seul (fond transparent) |
| `rbitnet-mark-on-ink.svg` | Avatar / favicon / fond sombre |
| `rbitnet-mark-on-mist.svg` | Fond clair / docs |
| `rbitnet-lockup.svg` | Mark + wordmark |
| `rbitnet-wordmark.svg` | Wordmark seul (**Rbit** ink + **net** mint) |
| `rbitnet-icon-512.png` | Icône App / OG / Discord |
| `favicon.svg` | Favicon site |
| `rbitnet-logo-ai-reference.png` | Référence illustrative (ne pas traiter comme source vectorielle) |
| `rbitnet-hero-atmosphere.png` | Visuel héro atmosphérique |

### Zone de protection

Laisser au moins **2 pixels logiques** (≈ 1/8 de la largeur du mark) autour du lapin. Ne pas poser de texte dans les oreilles. Ne pas déformer, ne pas ajouter d’ombre floue, ne pas recolorer hors palette.

### Versions

- **Fond clair :** mark transparent ou `on-mist`
- **Fond sombre :** `on-ink` ou mark blanc
- **Monochrome :** ink seul sur mist, ou blanc seul sur ink

## Couleurs

```css
:root {
  --ink: #0B1220;       /* texte, outlines pixel */
  --mist: #E8EEF4;      /* fond principal */
  --mist-deep: #D5DEE8; /* couches / grille */
  --paper: #F4F7FA;     /* surfaces légères */
  --mint: #00C896;      /* phosphor — accent marque */
  --amber: #F0A202;     /* bits / alerte douce */
  --fog: #9BB0C7;       /* lignes secondaires */
  --white: #F7FAFC;     /* fourrure pixel */
}
```

| Token | Hex | Rôle |
|-------|-----|------|
| Ink | `#0B1220` | Texte, contours |
| Mist | `#E8EEF4` | Fond atmosphère |
| Mint | `#00C896` | Accent primaire, « net », CTAs secondaires |
| Amber | `#F0A202` | Bits, highlights, CTAs primaires |
| Fog | `#9BB0C7` | Règles, meta, muted |

**À éviter :** violet / indigo gradient, crème + terracotta + serif display « AI default », néon cyberpunk multi-couches, pastilles arrondies empilées.

## Typographie

| Rôle | Famille | Usage |
|------|---------|--------|
| Display / marque | [Syne](https://fonts.google.com/specimen/Syne) 700–800 | Titres, hero **Rbitnet** |
| Corps | [Sora](https://fonts.google.com/specimen/Sora) 400–600 | Paragraphes, UI |
| Mono / bits | [IBM Plex Mono](https://fonts.google.com/specimen/IBM+Plex+Mono) | Wordmark optionnel, code, CLI |

Échelle indicative : hero marque ~clamp(3.5rem, 12vw, 7rem) ; H2 ~2rem ; corps 1.05–1.125rem ; line-height 1.5–1.65.

## Motion

Motions intentionnelles (pas de bruit) :

1. **Bit-grid drift** — grille de fond qui dérive très lentement
2. **Hero reveal** — marque + headline en fade/slide court (≤ 700 ms)
3. **Ear tick** — micro-animation du mark (1–2 px) au hover / idle

Réduire le mouvement si `prefers-reduced-motion: reduce`.

## Ton éditorial

- Direct, technique, sans marketing creux
- FR ou EN selon la surface ; le site vitrine peut être bilingue léger (FR lead)
- Verbes concrets : *mmap*, *GGUF*, *tok/s*, *OpenAI-compatible*
- Éviter : « révolutionnaire », « next-gen AI », emojis décoratifs

## Site vitrine

Source : [`website/`](../website/) — page unique pour GitHub Pages.  
Workflow : [`.github/workflows/pages.yml`](../.github/workflows/pages.yml).

## Checklist rapide

- [ ] Le viewport 1 sans nav reste identifiable **Rbitnet**
- [ ] Palette limitée aux tokens ci-dessus
- [ ] Mark pixel net (pas de redimensionnement flou — utiliser SVG ou multiples de la grille)
- [ ] Une intention par section
- [ ] CTA : Install / Docs / GitHub
