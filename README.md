# Kindle Capture

A macOS CLI tool that automatically captures Kindle book pages as PNG images. It uses screen capture and pixel hashing to reliably save every page, with support for resume, auto-detection of reading direction, and end-of-book detection.

## Requirements

- macOS (uses Quartz/CoreGraphics)
- Python 3.9+
- Kindle for Mac (`com.amazon.Lassen`)
- Screen Recording permission granted to your terminal app

## Quick Start

1. Open Kindle and navigate to the page where you want to start capturing.

2. Run the capture tool:

```bash
python3 kindle_capture.py
```

The tool will auto-detect the book title and reading direction, then capture each page sequentially.

## Usage

```
python3 kindle_capture.py [OPTIONS]

Options:
  --title TITLE          Book title (auto-detected if omitted)
  --output DIR           Output directory (default: ./captures)
  --direction {auto,rtl,ltr}
                         Reading direction (default: auto)
                           auto: probe keys automatically
                           rtl:  manga / vertical Japanese books
                           ltr:  horizontal / English books
  --delay SECONDS        Base delay after page turn (default: 0.8)
  --start-page N         Starting page number (auto-detected from existing files)
  --max-pages N          Max pages to capture (default: 5000)
  --end-retries N        Retries before declaring end-of-book (default: 8)
  --stable-timeout SECS  Timeout for frame stabilization (default: 8.0)
  --verbose              Print debug information
```

### Examples

```bash
# Capture a manga (right-to-left)
python3 kindle_capture.py --direction rtl

# Capture an English book with a custom title
python3 kindle_capture.py --direction ltr --title "My Book"

# Resume a previous capture session
python3 kindle_capture.py --title "My Book"
# (automatically detects existing pages and continues)
```

## How It Works

1. **Window detection** — Finds the Kindle window via Quartz CoreGraphics API.
2. **Capture method selection** — Tests three backends (CGImage per-window, `screencapture -l`, CGImage region crop) and selects the first that works.
3. **Navigation key probing** — Automatically determines which keys turn pages forward/backward using boundary detection or direction-independent keys (Space/PageDown).
4. **Pixel hash stabilization** — After each page turn, polls until two consecutive captures produce identical pixel hashes, ensuring the page has fully rendered.
5. **Duplicate detection** — Compares the new page hash against the previous page. If identical, retries with progressive backoff before declaring end-of-book.
6. **Resume** — On startup, scans the output directory for existing `p*.png` files and continues from the last page.

## E2E Diagnostic Tests

Run the test suite to verify your setup before capturing:

```bash
python3 test_e2e.py
python3 test_e2e.py --direction rtl   # force RTL
python3 test_e2e.py --verbose         # detailed output
```

The tests check: Kindle process, window detection, screenshot capture, pixel hash determinism, frame stabilization, navigation key probing, and page turn verification.

## Screen Recording Permission

The tool requires Screen Recording permission. On first run, if permission is not granted, you'll see instructions:

1. Open **System Settings > Privacy & Security > Screen Recording**
2. Enable your terminal app (Terminal, iTerm2, VS Code, etc.)
3. Restart the terminal app

## Output

Pages are saved as `p0001.png`, `p0002.png`, etc. in `./captures/<book-title>/`.

```
captures/
  My Book/
    p0001.png
    p0002.png
    ...
```

## License

MIT
