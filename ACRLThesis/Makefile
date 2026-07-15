MAIN = msc-thesis
PRES = msc-presentation

.PHONY: all pdf watch format clean distclean presentation presentation-notes

all: pdf

pdf:
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

# Defense slides, handout version (no notes).
presentation:
	latexmk -pdf -interaction=nonstopmode $(PRES).tex

# Presenter build: speaker notes on the right half of a dual-screen PDF.
presentation-notes:
	latexmk -pdf -interaction=nonstopmode $(PRES)-notes.tex

# Recompile on save.
watch:
	latexmk -pvc -pdf -interaction=nonstopmode $(MAIN).tex

# Reindents the main file plus every chapter/appendix file. Settings come
# from localSettings.yaml (tab indent, single rolling backup).
format:
	latexindent -w -m $(MAIN).tex chapters/*.tex

# Removes LaTeX build byproducts but keeps the PDF.
clean:
	latexmk -c $(MAIN).tex
	latexmk -c $(PRES).tex $(PRES)-notes.tex
	rm -f chapters/*.bak[0-9]* $(MAIN).bak[0-9]* indent.log chapters/indent.log

# Also removes the PDF itself.
distclean: clean
	latexmk -C $(MAIN).tex
	latexmk -C $(PRES).tex $(PRES)-notes.tex
