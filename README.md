# Modern Beamer Theme

A clean, self-sustained Beamer theme with modern colors and layout.

## Usage

- In your document preamble:
  - `\documentclass[11pt, aspectratio=169]{beamer}`
  - `\usetheme{Modern}`
  - Set `\title{...}`, `\author{...}`, `\date{...}`
- Render the title slide with `\maketitle` (no extra `\titlepage` frame needed).

## Compile

```sh
latexmk -pdf -interaction=nonstopmode -outdir=output template.tex
```

## Notes

- QR codes are disabled by default; enable via `\setboolean{Modern@useqrcodes}{true}` and define `\linktoslides` if you want a QR code.
- Footline shows short author (left), short title (center), and page number (right).
- The theme is fully self-contained in `beamerthemeModern.sty`.
