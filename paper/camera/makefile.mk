figs/shac.pdf: figs/shac.tex
	pdflatex --output-directory=figs --shell-escape figs/shac.tex

figs/shacpp.pdf: figs/shacpp.tex
	pdflatex --output-directory=figs --shell-escape figs/shacpp.tex

figs/attention.pdf: figs/attention.tex
	pdflatex --output-directory=figs --shell-escape figs/attention.tex

main.pdf: \
	figs/shac.pdf \
	figs/shacpp.pdf \
	figs/attention.pdf \
	main.tex sects/* tables/* appendices/* \
	local.bib
	pdflatex --shell-escape main.tex
	bibtex main
	bibtex bu1
	bibtex bu2
	pdflatex --shell-escape main.tex
	pdflatex --shell-escape main.tex

clean:
	rm -f main.aux 
	rm -f main.bbl
	rm -f main.blg
	rm -f main.log
	rm -f main.out
	rm -f main.pdf
	rm -f bu1.aux 
	rm -f bu1.bbl
	rm -f bu1.blg
	rm -f bu1.log
	rm -f bu1.out
	rm -f bu1.pdf
	rm -f bu2.aux 
	rm -f bu2.bbl
	rm -f bu2.blg
	rm -f bu2.log
	rm -f bu2.out
	rm -f bu2.pdf
	rm -f bu.aux 
	rm -f bu.bbl
	rm -f bu.blg
	rm -f bu.log
	rm -f bu.out
	rm -f bu.pdf
	rm -f figs/shac.pdf
	rm -f figs/shac.aux
	rm -f figs/shac.log
	rm -f figs/shac.out
	rm -f figs/shacpp.pdf
	rm -f figs/shacpp.aux
	rm -f figs/shacpp.log
	rm -f figs/shacpp.out
	rm -f figs/attention.pdf
	rm -f figs/attention.aux
	rm -f figs/attention.log

