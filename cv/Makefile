# Build CV PDF from LaTeX source
# Requires: latexmk (texlive-latex-extra)
# Optional RU: make ru (kept local; not published to site)

TEX = cv.tex
TEX_RU = cv_ru.tex
PDF = cv.pdf
PDF_RU = cv_ru.pdf
OUT = ../pdf/CV.pdf

.PHONY: all ru clean

all: $(OUT)

$(PDF): $(TEX)
	latexmk -pdf -interaction=nonstopmode $(TEX)

$(PDF_RU): $(TEX_RU)
	latexmk -pdf -interaction=nonstopmode $(TEX_RU)

$(OUT): $(PDF)
	cp $(PDF) $(OUT)

ru: $(PDF_RU)

clean:
	latexmk -pdf -C $(TEX)
	latexmk -pdf -C $(TEX_RU)
	rm -f $(OUT)
