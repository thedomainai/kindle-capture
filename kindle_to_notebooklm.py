#!/usr/bin/env python3
"""
Kindle to NotebookLM: Automate Kindle page capture, PDF generation, and NotebookLM import.

Usage:
    python kindle_to_notebooklm.py capture --book "Book Title" [--output-dir ./kindle_output] [--delay 2.5] [--max-pages 2000] [--resume]
    python kindle_to_notebooklm.py merge --input-dir ./kindle_output/BookTitle --output book.pdf
    python kindle_to_notebooklm.py upload --pdf book.pdf --notebook NOTEBOOK_ID
    python kindle_to_notebooklm.py run --book "Book Title" [--notebook NOTEBOOK_ID]

Requirements:
    - macOS with Kindle app installed (app name: "Amazon Kindle")
    - Python packages: Pillow, img2pdf
    - macOS permissions: Accessibility + Screen Recording for terminal app
    - For upload: notebooklm-mcp-cli (pip install notebooklm-mcp-cli)

Notes:
    - The tool uses fullscreen capture for reliability, as window ID detection
      via JXA/CoreGraphics is unreliable with the Kindle app in fullscreen mode.
    - Ensure Kindle is open with a book displayed before running capture.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
except ImportError:
    print("Missing dependency: Pillow. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "kindle_output"
DEFAULT_DELAY = 2.5
DEFAULT_MAX_PAGES = 2000
DEFAULT_SIMILARITY_THRESHOLD = 0.99
CONSECUTIVE_SAME_THRESHOLD = 3
NOTEBOOKLM_MAX_SIZE_MB = 200
PIXEL_TOLERANCE = 5


# ---------------------------------------------------------------------------
# PageComparator
# ---------------------------------------------------------------------------


class PageComparator:
    """Compare two screenshots to determine if they represent the same page."""

    def __init__(self, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.threshold = threshold

    def are_same(self, path_a: Path, path_b: Path) -> bool:
        """Return True if images are similar above the threshold."""
        similarity = self._compute_similarity(path_a, path_b)
        return similarity >= self.threshold

    @staticmethod
    def _compute_similarity(path_a: Path, path_b: Path) -> float:
        """Compute normalised pixel similarity between two grayscale images."""
        img_a = Image.open(path_a).convert("L")
        img_b = Image.open(path_b).convert("L")

        if img_a.size != img_b.size:
            return 0.0

        # Downsample for speed on Retina displays
        w, h = img_a.size
        if w > 1280:
            scale = 1280 / w
            new_size = (1280, int(h * scale))
            img_a = img_a.resize(new_size, Image.LANCZOS)
            img_b = img_b.resize(new_size, Image.LANCZOS)

        pixels_a = img_a.getdata()
        pixels_b = img_b.getdata()

        total = len(pixels_a)
        same = sum(
            1 for pa, pb in zip(pixels_a, pixels_b) if abs(pa - pb) <= PIXEL_TOLERANCE
        )
        return same / total


# ---------------------------------------------------------------------------
# KindleCapture
# ---------------------------------------------------------------------------


class KindleCapture:
    """Orchestrate Kindle app interaction and page screenshot capture."""

    def __init__(
        self,
        book_name: str,
        output_dir: Path,
        max_pages: int = DEFAULT_MAX_PAGES,
        delay: float = DEFAULT_DELAY,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        resume: bool = False,
    ):
        self.book_name = book_name
        self.output_dir = output_dir / self._sanitise_dirname(book_name)
        self.max_pages = max_pages
        self.delay = delay
        self.resume = resume
        self.comparator = PageComparator(similarity_threshold)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _sanitise_dirname(name: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', "_", name).strip()

    @staticmethod
    def _run_applescript(script: str) -> str:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"AppleScript error: {result.stderr.strip()}")
        return result.stdout.strip()

    @staticmethod
    def _run_jxa(script: str) -> str:
        result = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"JXA error: {result.stderr.strip()}")
        return result.stdout.strip()

    # -- Kindle interaction --------------------------------------------------

    def _activate_kindle(self) -> None:
        self._run_applescript('tell application "Amazon Kindle" to activate')
        time.sleep(1)

    def _enter_fullscreen(self) -> None:
        self._run_applescript(
            """
            tell application "System Events"
                tell process "Kindle"
                    keystroke "f" using {control down, command down}
                end tell
            end tell
            """
        )
        time.sleep(2)

    def _exit_fullscreen(self) -> None:
        # Pressing Escape exits fullscreen on most macOS apps
        self._run_applescript(
            """
            tell application "System Events"
                tell process "Kindle"
                    key code 53
                end tell
            end tell
            """
        )
        time.sleep(1)

    def _go_to_first_page(self) -> None:
        """Navigate to the first page of the book."""
        self._run_applescript(
            """
            tell application "System Events"
                tell process "Kindle"
                    key code 115 using {command down}
                end tell
            end tell
            """
        )
        time.sleep(self.delay)

    def _turn_page(self) -> None:
        """Advance one page forward."""
        self._run_applescript(
            """
            tell application "System Events"
                tell process "Kindle"
                    key code 124
                end tell
            end tell
            """
        )

    def _capture_page(self, page_num: int) -> Path:
        """Capture a fullscreen screenshot.

        Uses fullscreen capture instead of window-specific capture because
        JXA window ID detection is unreliable with Kindle in fullscreen mode.
        """
        out = self.output_dir / f"page_{page_num:04d}.png"
        subprocess.run(
            ["screencapture", "-x", "-o", str(out)],
            check=True,
            timeout=15,
        )
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"Screenshot failed for page {page_num}")
        return out

    # -- permission checks ---------------------------------------------------

    @staticmethod
    def check_permissions() -> bool:
        """Verify that Accessibility permission is granted."""
        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of first process whose frontmost is true',
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    # -- resume detection ----------------------------------------------------

    def _detect_existing_pages(self) -> int:
        """Return the last page number from existing files in output_dir."""
        if not self.output_dir.exists():
            return 0
        existing = sorted(self.output_dir.glob("page_*.png"))
        if not existing:
            return 0
        last = existing[-1].stem  # e.g. "page_0150"
        try:
            return int(last.split("_")[1])
        except (IndexError, ValueError):
            return 0

    # -- main capture loop ---------------------------------------------------

    def capture_all_pages(self) -> list[Path]:
        """Capture every page of the currently open Kindle book."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume support
        start_page = 1
        if self.resume:
            last_existing = self._detect_existing_pages()
            if last_existing > 0:
                print(f"Resuming from page {last_existing + 1} (found {last_existing} existing pages)")
                start_page = last_existing + 1

        # Permission check
        if not self.check_permissions():
            print(
                "ERROR: Accessibility permission not granted.\n"
                "Go to System Settings > Privacy & Security > Accessibility\n"
                "and add your terminal application.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Activating Kindle for '{self.book_name}'...")
        self._activate_kindle()

        print("Entering fullscreen...")
        self._enter_fullscreen()
        time.sleep(1)

        if start_page == 1:
            print("Navigating to first page...")
            self._go_to_first_page()
        else:
            # For resume, skip to the right page by turning pages
            print(f"Skipping to page {start_page}...")
            # Pages are already captured; just need to turn past them
            for _ in range(start_page - 1):
                self._turn_page()
                time.sleep(0.3)
            time.sleep(self.delay)

        captured_pages: list[Path] = []

        # Include existing pages if resuming
        if self.resume and start_page > 1:
            for i in range(1, start_page):
                p = self.output_dir / f"page_{i:04d}.png"
                if p.exists():
                    captured_pages.append(p)

        prev_image_path: Optional[Path] = None
        consecutive_same = 0

        print(f"Starting capture (delay={self.delay}s, max={self.max_pages} pages)...")
        print("Press Ctrl+C to stop early.\n")

        try:
            for page_num in range(start_page, self.max_pages + 1):
                image_path = self._capture_page(page_num)

                # Last-page detection
                if prev_image_path is not None:
                    if self.comparator.are_same(prev_image_path, image_path):
                        consecutive_same += 1
                        if consecutive_same >= CONSECUTIVE_SAME_THRESHOLD:
                            # Remove duplicate captures
                            for dup_num in range(
                                page_num - consecutive_same + 1, page_num + 1
                            ):
                                dup_path = self.output_dir / f"page_{dup_num:04d}.png"
                                dup_path.unlink(missing_ok=True)
                            captured_pages = captured_pages[: -(consecutive_same - 1)]
                            actual_last = page_num - consecutive_same
                            print(f"\nLast page detected: page {actual_last}")
                            break
                    else:
                        consecutive_same = 0

                captured_pages.append(image_path)
                prev_image_path = image_path

                # Progress
                print(f"  Page {page_num} captured", end="")
                if consecutive_same > 0:
                    print(f" (same={consecutive_same}/{CONSECUTIVE_SAME_THRESHOLD})", end="")
                print()

                # Turn page
                self._turn_page()
                time.sleep(self.delay)
            else:
                print(f"\nWARNING: Reached max pages ({self.max_pages}). Book may have more pages.")

        except KeyboardInterrupt:
            print(f"\n\nInterrupted. Captured {len(captured_pages)} pages so far.")
            print("Use --resume to continue later.")

        print(f"\nCapture complete: {len(captured_pages)} pages in {self.output_dir}")
        return captured_pages


# ---------------------------------------------------------------------------
# PdfMerger
# ---------------------------------------------------------------------------


class PdfMerger:
    """Merge page images into a single PDF."""

    @staticmethod
    def merge(image_paths: list[Path], output_path: Path) -> Path:
        """Combine ordered PNG files into a single PDF."""
        sorted_paths = sorted(image_paths, key=lambda p: p.name)

        for p in sorted_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing image: {p}")

        if not sorted_paths:
            raise ValueError("No images to merge")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try img2pdf first (lossless), fallback to Pillow
        try:
            import img2pdf as _img2pdf

            pdf_bytes = _img2pdf.convert([str(p) for p in sorted_paths])
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
        except ImportError:
            print("img2pdf not available, using Pillow (may re-compress images)...")
            images = [Image.open(p).convert("RGB") for p in sorted_paths]
            images[0].save(
                output_path, save_all=True, append_images=images[1:], format="PDF"
            )

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"PDF created: {output_path} ({len(sorted_paths)} pages, {size_mb:.1f} MB)")

        if size_mb > NOTEBOOKLM_MAX_SIZE_MB:
            print(
                f"WARNING: PDF is {size_mb:.0f} MB, exceeding NotebookLM's {NOTEBOOKLM_MAX_SIZE_MB} MB limit.",
                file=sys.stderr,
            )

        return output_path

    @staticmethod
    def merge_from_dir(input_dir: Path, output_path: Path) -> Path:
        """Merge all page_*.png files in a directory."""
        pages = sorted(input_dir.glob("page_*.png"))
        if not pages:
            raise FileNotFoundError(f"No page_*.png files found in {input_dir}")
        return PdfMerger.merge(pages, output_path)


# ---------------------------------------------------------------------------
# NotebookLMUploader
# ---------------------------------------------------------------------------


class NotebookLMUploader:
    """Upload PDF to NotebookLM via notebooklm-mcp-cli."""

    @staticmethod
    def _find_nlm() -> Optional[str]:
        """Find the nlm CLI command."""
        path = shutil.which("nlm")
        if path:
            return path
        # Try within venv
        venv_nlm = Path(__file__).parent.parent.parent / ".venv" / "bin" / "nlm"
        if venv_nlm.exists():
            return str(venv_nlm)
        return None

    @staticmethod
    def check_cli() -> bool:
        """Check if nlm CLI is installed."""
        return NotebookLMUploader._find_nlm() is not None

    @staticmethod
    def upload(pdf_path: Path, notebook_id: str) -> bool:
        """Upload a PDF as a source to a NotebookLM notebook."""
        nlm = NotebookLMUploader._find_nlm()
        if not nlm:
            print(
                "ERROR: nlm CLI not found. Install with: pip install notebooklm-mcp-cli",
                file=sys.stderr,
            )
            return False

        if not pdf_path.exists():
            print(f"ERROR: PDF file not found: {pdf_path}", file=sys.stderr)
            return False

        print(f"Uploading {pdf_path.name} to NotebookLM notebook {notebook_id}...")
        result = subprocess.run(
            [nlm, "source", "add", notebook_id, "--file", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print(f"Upload successful.")
            if result.stdout.strip():
                print(result.stdout.strip())
            return True
        else:
            print(f"Upload failed: {result.stderr.strip()}", file=sys.stderr)
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_capture(args: argparse.Namespace) -> list[Path]:
    """Handle the 'capture' subcommand."""
    capture = KindleCapture(
        book_name=args.book,
        output_dir=Path(args.output_dir),
        max_pages=args.max_pages,
        delay=args.delay,
        similarity_threshold=args.similarity_threshold,
        resume=args.resume,
    )
    return capture.capture_all_pages()


def cmd_merge(args: argparse.Namespace) -> Path:
    """Handle the 'merge' subcommand."""
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    return PdfMerger.merge_from_dir(input_dir, output)


def cmd_upload(args: argparse.Namespace) -> bool:
    """Handle the 'upload' subcommand."""
    return NotebookLMUploader.upload(Path(args.pdf), args.notebook)


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand (full pipeline)."""
    # Step 1: Capture
    capture = KindleCapture(
        book_name=args.book,
        output_dir=Path(args.output_dir),
        max_pages=args.max_pages,
        delay=args.delay,
        similarity_threshold=args.similarity_threshold,
        resume=args.resume,
    )
    pages = capture.capture_all_pages()

    if not pages:
        print("No pages captured. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Merge
    sanitised = re.sub(r'[<>:"/\\|?*]', "_", args.book).strip()
    pdf_path = capture.output_dir / f"{sanitised}.pdf"
    PdfMerger.merge(pages, pdf_path)

    # Step 3: Upload (optional)
    if args.notebook:
        NotebookLMUploader.upload(pdf_path, args.notebook)
    else:
        print(f"\nPDF ready at: {pdf_path}")
        print("To upload to NotebookLM, run:")
        print(f'  python {__file__} upload --pdf "{pdf_path}" --notebook YOUR_NOTEBOOK_ID')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kindle_to_notebooklm",
        description="Automate Kindle page capture, PDF generation, and NotebookLM import.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- capture --
    p_cap = sub.add_parser("capture", help="Capture all Kindle pages as screenshots")
    p_cap.add_argument("--book", required=True, help="Book title (used as folder name)")
    p_cap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Base output directory")
    p_cap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between page captures")
    p_cap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Maximum pages to capture")
    p_cap.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Threshold for same-page detection (0.0-1.0)",
    )
    p_cap.add_argument("--resume", action="store_true", help="Resume from last captured page")

    # -- merge --
    p_merge = sub.add_parser("merge", help="Merge page screenshots into a PDF")
    p_merge.add_argument("--input-dir", required=True, help="Directory containing page_*.png files")
    p_merge.add_argument("--output", required=True, help="Output PDF path")

    # -- upload --
    p_upload = sub.add_parser("upload", help="Upload PDF to NotebookLM")
    p_upload.add_argument("--pdf", required=True, help="Path to PDF file")
    p_upload.add_argument("--notebook", required=True, help="NotebookLM notebook ID")

    # -- run (full pipeline) --
    p_run = sub.add_parser("run", help="Full pipeline: capture + merge + upload")
    p_run.add_argument("--book", required=True, help="Book title")
    p_run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Base output directory")
    p_run.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between page captures")
    p_run.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Maximum pages to capture")
    p_run.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Threshold for same-page detection (0.0-1.0)",
    )
    p_run.add_argument("--resume", action="store_true", help="Resume from last captured page")
    p_run.add_argument("--notebook", default=None, help="NotebookLM notebook ID (optional)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "capture":
        cmd_capture(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "upload":
        cmd_upload(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
