"""GUI entry point: ``python -m vacuum`` / ``vacuum``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from vacuum.ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vacuum robot console")
    parser.add_argument("--host", default="localhost", help="simulator host")
    parser.add_argument("--port-base", type=int, default=5555, help="first simulator port")
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("recordings"),
        help="where camera recordings are written",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Vacuum Robot Console")
    window = MainWindow(args.host, args.port_base, args.recordings_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
