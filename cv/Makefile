# Build CV PDF from LaTeX source
# Requires: latexmk (texlive), node (for BibTeX → publications.tex)
# Optional RU: make ru (kept local; not published to site)

TEX = cv.tex
TEX_RU = cv_ru.tex
PUBS = publications.tex
PDF = cv.pdf
PDF_RU = cv_ru.pdf
OUT = ../pdf/CV.pdf
BIB = ../content/publications.bib
GEN = ../scripts/generate-cv-pubs.js

.PHONY: all pubs ru clean

all: $(OUT)

pubs: $(PUBS)

$(PUBS): $(BIB) $(GEN) ../src/_data/publications.js
	node $(GEN)

$(PDF): $(TEX) $(PUBS)
	latexmk -pdf -interaction=nonstopmode $(TEX)

$(PDF_RU): $(TEX_RU)
	latexmk -pdf -interaction=nonstopmode $(TEX_RU)

$(OUT): $(PDF)
	cp $(PDF) $(OUT)

ru: $(PDF_RU)

clean:
	latexmk -pdf -C $(TEX)
	latexmk -pdf -C $(TEX_RU)
	rm -f $(OUT) $(PUBS)
