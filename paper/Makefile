# GraphPalace Paper — Build with pdflatex
#
# Usage:
#   make          — build PDF (two pdflatex passes for references)
#   make once     — single pdflatex pass (faster, no cross-refs)
#   make clean    — remove build artefacts (keep PDF)
#   make distclean — remove everything including PDF
#   make view     — open PDF in system viewer (Linux/macOS)
#   make watch    — rebuild on file change (requires entr or inotifywait)

PAPER   = graphpalace-paper
LATEX   = pdflatex
LATEXFLAGS = -interaction=nonstopmode -halt-on-error
BIBTEX  = bibtex

.PHONY: all once clean distclean view watch

all: $(PAPER).pdf

# Two-pass build: first pass generates aux, second resolves references
$(PAPER).pdf: $(PAPER).tex
	$(LATEX) $(LATEXFLAGS) $(PAPER).tex
	$(LATEX) $(LATEXFLAGS) $(PAPER).tex
	@echo ""
	@echo "=== Build complete: $(PAPER).pdf ==="

once: $(PAPER).tex
	$(LATEX) $(LATEXFLAGS) $(PAPER).tex
	@echo ""
	@echo "=== Single pass complete: $(PAPER).pdf ==="

clean:
	rm -f $(PAPER).aux \
	       $(PAPER).log \
	       $(PAPER).out \
	       $(PAPER).toc \
	       $(PAPER).bbl \
	       $(PAPER).blg \
	       $(PAPER).lof \
	       $(PAPER).lot \
	       $(PAPER).lol \
	       $(PAPER).fls \
	       $(PAPER).fdb_latexmk \
	       $(PAPER).synctex.gz \
	       texput.log
	@echo "Cleaned build artefacts."

distclean: clean
	rm -f $(PAPER).pdf
	@echo "Removed PDF."

# Open with system PDF viewer (adjust for your OS)
view: $(PAPER).pdf
	@if command -v evince > /dev/null; then evince $(PAPER).pdf &; \
	elif command -v okular > /dev/null; then okular $(PAPER).pdf &; \
	elif command -v xdg-open > /dev/null; then xdg-open $(PAPER).pdf &; \
	elif command -v open > /dev/null; then open $(PAPER).pdf; \
	else echo "No PDF viewer found. Open $(PAPER).pdf manually."; fi

# Watch for changes and rebuild (requires entr: apt-get install entr)
watch:
	@if command -v entr > /dev/null; then \
	  echo "Watching $(PAPER).tex for changes (Ctrl-C to stop)..."; \
	  echo $(PAPER).tex | entr -c make once; \
	else \
	  echo "entr not found. Install with: apt-get install entr"; \
	  exit 1; \
	fi
