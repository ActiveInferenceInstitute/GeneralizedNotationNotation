# LaTeX Preamble

This file contains project-local LaTeX additions consumed by the template renderer.
Keep it minimal until the manuscript needs additional math, table, or figure support.

Every package below ships with a base TeX Live installation. Anything outside that
set must be loaded through the `\IfFileExists` guard the template itself uses for
`seqsplit` (see `output/pdf/_combined_manuscript.tex`), so a missing package
degrades instead of failing the build:

```
\IfFileExists{pkg.sty}{\usepackage{pkg}}{<fallback definition>}
```

`cleveref` was loaded here unguarded until 2026-09-05. It resolved only from a
hand-installed personal `texmf` tree (`kpsewhich cleveref.sty` →
`~/Library/texmf/...`, an eight-year-old copy that triggered LaTeX's "First Aid"
patch), it is absent from the `texlive-basic` scheme, and the document contains
zero `\cref`/`\Cref` calls — every cross-reference goes through pandoc-crossref,
which emits plain `\ref`. Reinstating it means guarding it AND converting the
`\ref` calls.

```latex
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{hyperref}
\usepackage{natbib}

% Contents listing: article.cls prefixes every \l@section entry with
% \addvspace{1.0em}, which across ten sections costs about ten lines and pushed
% the ~40-entry listing one line past page 2 — leaving a third sheet carrying a
% single "10 References" line. Tighten that inter-section gap for the duration
% of \tableofcontents only, inside a group, so body spacing is untouched and no
% entry is dropped (reducing \tocdepth would have hidden 29 real subsections).
\makeatletter
\let\GNNoriginaltableofcontents\tableofcontents
\renewcommand{\tableofcontents}{%
  \begingroup
  \setlength{\parskip}{0pt}%
  \renewcommand{\addvspace}[1]{\vskip 2\p@}%
  \GNNoriginaltableofcontents
  \endgroup}
\makeatother
```
