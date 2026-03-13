"""Shared utility functions for real-world demo pipeline."""

import json
from pathlib import Path


class NotebookParseError(Exception):
    """Raised when a Jupyter notebook cannot be parsed."""

    pass


def extract_code_from_notebook(notebook_path: Path) -> str:
    """Extract Python code from Jupyter notebook.

    Parses the notebook JSON structure and concatenates all code cell sources.
    This is more efficient than passing raw JSON to LLM (saves tokens).

    Args:
        notebook_path: Path to .ipynb file.

    Returns:
        Concatenated Python code from all code cells, separated by blank lines.

    Raises:
        NotebookParseError: If notebook cannot be parsed.
    """
    try:
        with open(notebook_path, encoding="utf-8") as f:
            notebook = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise NotebookParseError(f"Could not parse notebook {notebook_path}: {e}") from e

    code_cells = []
    cells = notebook.get("cells", [])

    for cell in cells:
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                code_cells.append("".join(source))
            elif isinstance(source, str):
                code_cells.append(source)

    return "\n\n".join(code_cells)
