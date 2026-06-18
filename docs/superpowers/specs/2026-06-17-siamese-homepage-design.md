# Siamese PyTorch GitHub Pages Homepage Design

Date: 2026-06-17
Status: Approved for planning

## Goal

Create a visually strong project homepage for `fangpin/siamese-pytorch` using GitHub Pages. The page must:

- default to English
- support English and Chinese switching on the page
- feel modern and technically polished
- balance project presentation with practical developer information
- expose the GitHub repository clearly in the header, hero, and footer

## Project Context

This repository is a compact PyTorch reimplementation of Siamese Networks for one-shot learning on the Omniglot dataset. The current repo shape matters:

- no frontend toolchain or `package.json`
- core source files live at the repo root: `train.py`, `model.py`, `mydataset.py`, `make_dataset.py`
- the existing README already contains the essential run commands and experiment summary
- the repo already includes a useful visual asset: `loss.png`
- the repository remote is `git@github.com:fangpin/siamese-pytorch.git`

The homepage should respect that reality instead of pretending this is a large product site.

## Approaches Considered

### 1. Research Signal

A dark, cinematic hero-led page with the model identity presented as a visual system first.

- strongest visual punch
- weaker immediate technical scannability

### 2. Paper Wall

An editorial, notebook-style layout that foregrounds paper context, results, and implementation notes.

- strongest research credibility
- less visually distinctive than the darker directions

### 3. Model Console

A dark technical interface with expressive hero treatment and compact information cards.

- strongest balance of cool factor and technical usefulness
- maps cleanly to a static site without a JS framework

## Chosen Direction

Use **Model Console**.

This direction matches the stated goal: the homepage should feel presentational first, but still become useful to developers immediately after the hero. The visual identity should be dark, sharp, and technical rather than editorial or marketing-heavy.

## Delivery Architecture

Implement the site as a **pure static GitHub Pages site** with no framework and no build step for the page itself.

### Site source

- create a dedicated `site/` directory for the homepage
- keep the Python project files untouched at the repo root
- place page assets under `site/assets/`

### GitHub Pages deployment

- add a minimal GitHub Actions Pages workflow under `.github/workflows/pages.yml`
- deploy the static contents of `site/`
- target the current default branch workflow path rather than relying on `docs/` publishing

This keeps the site isolated from the Python source tree and avoids exposing `docs/superpowers/specs/` as part of the public site.

## Page Structure

The homepage remains a single scrollable page with the following sections, in this order:

1. **Top bar**
2. **Hero**
3. **Quick facts / metrics**
4. **Architecture snapshot**
5. **Quick start**
6. **Implementation notes**
7. **Training curve**
8. **Footer links**

### 1. Top bar

The top bar includes:

- project name
- short subtitle
- explicit GitHub link using the real repo identity: `fangpin/siamese-pytorch`
- English / Chinese toggle

The GitHub link is a primary navigation item, not a hidden footer-only link.

### 2. Hero

The hero introduces the project with:

- a concise headline about Siamese one-shot learning
- a short supporting description grounded in Omniglot and PyTorch
- primary CTA: `GitHub Repository`
- secondary CTA: `Quick Start`

The hero should visually lead the page, but it must not become a decorative landing-page block detached from the actual project.

### 3. Quick facts / metrics

Use compact cards to surface the strongest scan-level facts from the repo:

- `89.5%` final accuracy
- `20-way` one-shot evaluation
- `PyTorch` implementation
- `Omniglot` dataset

These cards help visitors understand the project without reading paragraphs first.

### 4. Architecture snapshot

This section visualizes the current network structure from `model.py` in a compact schematic form. It should reflect the real implementation, not a generic diagram:

- stacked convolution blocks
- shared twin encoder concept
- absolute difference comparison
- final linear output layer

The section is explanatory, not interactive.

### 5. Quick start

Surface the actual repo workflow from the README:

- clone / prepare Omniglot data
- unzip evaluation and background sets
- create the model output directory
- run the provided training command

This section should include a well-formatted code block and short context labels, not a prose-only explanation.

### 6. Implementation notes

Summarize the two main differences from the original paper already stated in the README:

- Adam instead of SGD with momentum
- default PyTorch parameter settings instead of layer-specific initialization / regularization choices

This section should also mention why the reported result is slightly below the paper number.

### 7. Training curve

Reuse the existing `loss.png` as the main experiment artifact. The section should frame it as evidence of the training run rather than decorative media.

### 8. Footer links

Footer links must include:

- GitHub repository
- original paper
- Omniglot dataset
- source files: `train.py`, `model.py`, `mydataset.py`

The footer is the third explicit GitHub entry point on the page.

## Visual System

### Tone

The visual language should feel like a model dashboard rather than a generic SaaS template or a paper PDF clone.

### Color

Use a dark base with cool blue-cyan accents and a restrained warm highlight for emphasis. Avoid purple-heavy gradient clichés.

### Typography

Use a technical but distinctive pairing:

- display / UI heading font with machine-like character
- clean readable body font
- monospace for commands and architecture labels

Typography must stay controlled on mobile and never rely on viewport-based font scaling.

### Surfaces

Use paneled surfaces, subtle grid texture, crisp borders, restrained glow, and small-radius cards. Avoid giant floating cards and soft blob decoration.

### Motion

Use lightweight animation only:

- hero reveal
- subtle card hover
- small chart / signal motion if implemented with CSS only

Motion should support polish, not dominate the page.

## Bilingual Behavior

The page defaults to English.

### Language switch behavior

- header toggle with `EN` and `中文`
- English selected on first load
- store the user choice in `localStorage`
- restore the saved choice on later visits
- update all translatable copy in place without navigation
- update the root `lang` attribute to match the current language

### Content model

Implement translations through a small JavaScript dictionary keyed by stable content ids. The HTML structure remains shared; only the content values swap.

### What does not change across languages

- code blocks
- metric values
- external links
- source file names

## Responsive Behavior

### Desktop

- hero uses a split layout: narrative left, utility panels right
- lower sections use 2-column or compact grid layouts where appropriate

### Mobile

- all sections collapse into a single column
- navigation controls wrap cleanly
- CTA buttons remain stable in size
- code blocks scroll horizontally if needed
- the architecture snapshot simplifies spacing without hiding meaning

No text or controls may overlap at narrow widths.

## Accessibility and Semantics

- use semantic sections, headings, nav, main, and footer
- language toggle uses real buttons with visible active state
- links remain clearly identifiable
- color contrast must hold on the dark background
- images include meaningful alt text

## Files to Add or Modify

Planned implementation footprint:

- `.github/workflows/pages.yml`
- `site/index.html`
- `site/assets/site.css`
- `site/assets/site.js`
- `site/assets/loss.png` or a direct reused asset path if cleaner

The existing Python files should remain functionally unchanged.

## Verification Plan

Before claiming completion, verify:

- the static site renders locally
- the language toggle defaults to English and persists after switching
- all GitHub, paper, dataset, and source links resolve correctly
- the layout remains coherent on desktop and mobile widths
- the Pages workflow file is syntactically valid

## Out of Scope

- turning the repo into a React or Vite app
- adding backend services
- changing the training code or model behavior
- rewriting the README beyond linking to it from the page
