# Build CV PDFs from LaTeX source
# Requires: latexmk (texlive-latex-extra, texlive-lang-cyrillic)

TEX = cv.tex
TEX_RU = cv_ru.tex
PDF = cv.pdf
PDF_RU = cv_ru.pdf
OUT = ../pdf/CV.pdf
OUT_RU = ../pdf/CV_ru.pdf

.PHONY: all clean

all: $(OUT) $(OUT_RU)

$(PDF): $(TEX)
	latexmk -pdf -interaction=nonstopmode $(TEX)

$(PDF_RU): $(TEX_RU)
	latexmk -pdf -interaction=nonstopmode $(TEX_RU)

$(OUT): $(PDF)
	cp $(PDF) $(OUT)

$(OUT_RU): $(PDF_RU)
	cp $(PDF_RU) $(OUT_RU)

clean:
	latexmk -pdf -C $(TEX)
	latexmk -pdf -C $(TEX_RU)
	rm -f $(OUT) $(OUT_RU)
