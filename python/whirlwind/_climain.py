"""Console entry point for the ``whirlwind`` CLI (``[project.scripts]``).

A thin shim over the Rust CLI (``whirlwind_cli::run`` via ``_native.cli_main``)
so pip/uvx installs get the executable without a second implementation or
distribution - e.g. ``uvx --from whirlwind-insar whirlwind --help``. The
prebuilt binaries attached to each GitHub Release remain the zero-Python path,
sharing the exact same Rust code.
"""

import sys

from ._native import cli_main


def main() -> None:
    sys.exit(cli_main(sys.argv[1:]))
