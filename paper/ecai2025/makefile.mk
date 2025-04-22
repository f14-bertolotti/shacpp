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
	pdflatex --shell-escape main.tex
	pdflatex --shell-escape main.tex

clean:
	rm -f main.aux 
	rm -f main.bbl
	rm -f main.blg
	rm -f main.log
	rm -f main.out
	rm -f main.pdf
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

