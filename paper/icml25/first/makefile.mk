
main.pdf: main.tex sects/* local.bib
	pdflatex --shell-escape main.tex
	bibtex main
	pdflatex --shell-escape main.tex
	pdflatex --shell-escape main.tex

clean:
	rm -f main.aux 
	rm -f main.bbl
	rm -f main.blg
	rm -f main.log
	rm -f main.out
	rm -f main.pdf


