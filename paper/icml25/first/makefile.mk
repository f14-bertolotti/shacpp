
main.aux: main.tex
	pdflatex --shell-escape main.tex

main.bbl: main.aux
	bibtex main

main.pdf: main.aux main.bbl
	pdflatex --shell-escape main.tex

clean:
	rm -f main.aux 
	rm -f main.bbl
	rm -f main.blg
	rm -f main.log
	rm -f main.out
	rm -f main.pdf


