
MY_TEX_SRC=msc-thesis.tex
MAIN_FILE=msc-thesis
MY_BIB_SRC=references.bib

all: msc-thesis.pdf


msc-thesis.pdf: $(MY_TEX_SRC) $(MY_BIB_SRC)
	pdflatex $(MAIN_FILE)
	bibtex $(MAIN_FILE)
	pdflatex $(MAIN_FILE)
	pdflatex $(MAIN_FILE)

clean:
	rm -f $(MAIN_FILE).aux $(MAIN_FILE).bbl $(MAIN_FILE).blg $(MAIN_FILE).log \
	      $(MAIN_FILE).out $(MAIN_FILE).toc $(MAIN_FILE).lof $(MAIN_FILE).lot \
	      $(MAIN_FILE).glo $(MAIN_FILE).gls $(MAIN_FILE).glg $(MAIN_FILE).ist \
	      $(MAIN_FILE).acn $(MAIN_FILE).acr $(MAIN_FILE).alg

.PHONY: all clean

