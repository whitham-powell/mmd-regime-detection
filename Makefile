.PHONY: env test plots md clean

env:			## (Re)sync venv with dev extras
	uv pip install -e '.[dev]'

add=%  # make add name=pandas
add:
	uv add $(name)

test:			## Run pytest under uv
	uv run pytest -q

plots: env		## Extract plots from executed notebooks
	@mkdir -p extracted_figures
	@for nb in notebooks/*.ipynb notebooks/**/*.ipynb; do \
		if [ -f "$$nb" ]; then \
			echo "Executing $$nb..."; \
			uv run jupyter nbconvert "$$nb" \
				--to html \
				--execute \
				--ExecutePreprocessor.timeout=600 \
				--ExtractOutputPreprocessor.enabled=True 2>/dev/null || true; \
		fi \
	done
	@find . -type d -name "*_files" | while read dir; do \
		cp $$dir/* extracted_figures/ 2>/dev/null || true; \
		rm -r $$dir; \
	done
	@find notebooks -name '*.html' -delete 2>/dev/null || true
	@echo "Plots extracted to extracted_figures/"

md: env			## Convert notebooks to markdown
	@mkdir -p notebooks/markdown_exports
	@for nb in notebooks/*.ipynb notebooks/**/*.ipynb; do \
		if [ -f "$$nb" ]; then \
			echo "Converting $$nb to Markdown..."; \
			uv run jupyter nbconvert "$$nb" \
				--to markdown \
				--execute \
				--ExecutePreprocessor.timeout=600 \
				--output-dir=notebooks/markdown_exports 2>/dev/null || true; \
		fi \
	done

sync:			## Sync .py <-> .ipynb via jupytext
	uv run jupytext --sync notebooks/*.py notebooks/**/*.py 2>/dev/null || true

clean:			## Remove generated files
	rm -rf extracted_figures/
	rm -rf notebooks/markdown_exports/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete
	find . -name ".pytest_cache" -type d -delete

help:			## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
