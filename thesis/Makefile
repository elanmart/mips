all:
	pdflatex -shell-escape thesis.tex
	bibtex thesis
	pdflatex -shell-escape thesis.tex
graph:
	mkdir ../graphs
	cd .. && python3 result_parser.py
clean:
	rm -r ../graphs
	rm *.aux *.bbl *.blg *.dvi *.log *.out *.toc
