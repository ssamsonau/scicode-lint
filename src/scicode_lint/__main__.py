"""Entry point for python -m scicode_lint."""

import sys

from scicode_lint.cli import main

if __name__ == "__main__":
    sys.exit(main())
