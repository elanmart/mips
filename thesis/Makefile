all: ../graphs/nondominated/Amazon-3M-p1.eps
	pdflatex -shell-escape thesis.tex
	bibtex thesis
	pdflatex -shell-escape thesis.tex

graph: ../graphs/nondominated/Amazon-3M-p1.eps

../graphs/nondominated/Amazon-3M-p1.eps: ../result_parser.py
	#mkdir ../graphs
	cd .. && python3 result_parser.py

clean:
	rm -rf ../graphs
	rm -f *.aux *.bbl *.blg *.dvi *.log *.out *.toc
