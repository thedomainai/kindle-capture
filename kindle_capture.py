#!/usr/bin/env python3
"""Kindle Capture — macOS CLI tool to capture Kindle book pages as PNG images."""

import argparse
import glob as _glob
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time

try:
    import Quartz.CoreGraphics as CG
    import Quartz
except ImportError:
    print("Error: Quartz framework not available. This tool requires macOS.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Screen recording permission check
# ---------------------------------------------------------------------------

def _detect_host_app():
    """Detect the host application that needs screen recording permission."""
    term = os.environ.get("TERM_PROGRAM", "")
    if term == "vscode":
        return "Visual Studio Code"
    if term == "iTerm.app":
        return "iTerm2"
    if term == "Apple_Terminal":
        return "Terminal"
    if term == "WarpTerminal":
        return "Warp"
    # Fallback: check parent process
    try:
        result = subprocess.run(
            ["ps", "-p", str(os.getppid()), "-o", "comm="],
            capture_output=True, text=True,
        )
        comm = result.stdout.strip()
        if comm:
            return os.path.basename(comm)
    except Exception:
        pass
    return "your terminal app"


def check_screen_recording_permission():
    """Test if screen recording is permitted by attempting a capture.

    Returns True if permitted, False otherwise.
    """
    # Method 1: Try CGWindowListCreateImage on any visible window
    img = CG.CGWindowListCreateImage(
        CG.CGRectInfinite,
        CG.kCGWindowListOptionOnScreenOnly,
        CG.kCGNullWindowID,
        CG.kCGWindowImageDefault,
    )
    if img is not None:
        w = CG.CGImageGetWidth(img)
        h = CG.CGImageGetHeight(img)
        # Without permission, macOS may return a 1x1 transparent image
        if w > 1 and h > 1:
            return True

    # Method 2: Try screencapture command
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        subprocess.run(
            ["screencapture", "-x", tmp],
            capture_output=True, timeout=5,
        )
        if os.path.exists(tmp) and os.path.getsize(tmp) > 100:
            return True
    except Exception:
        pass
    finally:
        _safe_remove(tmp)

    return False


# ---------------------------------------------------------------------------
# Window detection
# ---------------------------------------------------------------------------

def find_kindle_window():
    """Find the main Kindle window (largest layer-0 window by area)."""
    window_list = CG.CGWindowListCopyWindowInfo(
        CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID
    )
    best = None
    best_area = 0
    for w in window_list:
        if w.get("kCGWindowOwnerName", "") != "Kindle":
            continue
        if w.get("kCGWindowLayer", -1) != 0:
            continue
        bounds = w.get("kCGWindowBounds", {})
        width = int(bounds.get("Width", 0))
        height = int(bounds.get("Height", 0))
        area = width * height
        if area < 10000:
            continue
        if area > best_area:
            best_area = area
            best = {
                "id": w.get("kCGWindowNumber"),
                "name": w.get("kCGWindowName", ""),
                "x": int(bounds.get("X", 0)),
                "y": int(bounds.get("Y", 0)),
                "width": width,
                "height": height,
            }
    return best


def is_kindle_running():
    """Check if Kindle process is running."""
    result = subprocess.run(["pgrep", "-x", "Kindle"], capture_output=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Screenshot — three backends with auto-selection
#   1. CGWindowListCreateImage per-window (fastest, may fail on macOS 26+)
#   2. screencapture -l (subprocess, may fail on macOS 26+)
#   3. CGWindowListCreateImage region crop (captures entire screen, crops to
#      window bounds — works when per-window capture is blocked)
# ---------------------------------------------------------------------------

def _save_cgimage_to_png(cgimage, output_path):
    """Save a CGImage to a PNG file. Returns True on success."""
    url = Quartz.CFURLCreateWithFileSystemPath(
        None, output_path, Quartz.kCFURLPOSIXPathStyle, False
    )
    if url is None:
        return False
    dest = CG.CGImageDestinationCreateWithURL(url, "public.png", 1, None)
    if dest is None:
        return False
    CG.CGImageDestinationAddImage(dest, cgimage, None)
    return bool(CG.CGImageDestinationFinalize(dest))


def _capture_via_cgimage(window_id, output_path):
    """Capture using CGWindowListCreateImage per-window API."""
    img = CG.CGWindowListCreateImage(
        CG.CGRectNull,
        CG.kCGWindowListOptionIncludingWindow,
        window_id,
        CG.kCGWindowImageBoundsIgnoreFraming,
    )
    if img is None:
        return False
    w = CG.CGImageGetWidth(img)
    h = CG.CGImageGetHeight(img)
    if w < 10 or h < 10:
        return False
    return _save_cgimage_to_png(img, output_path)


def _capture_via_screencapture(window_id, output_path):
    """Capture using screencapture -l command."""
    subprocess.run(
        ["screencapture", "-l", str(window_id), "-x", "-o", output_path],
        capture_output=True,
    )
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def _capture_via_region(window_id, output_path):
    """Capture the screen region where the window is located.

    This works even when per-window capture is blocked (macOS 26+).
    It captures all on-screen content at the window's bounds, so the
    window must be frontmost and unobstructed.
    """
    # Get fresh window bounds
    window_list = CG.CGWindowListCopyWindowInfo(
        CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID
    )
    bounds = None
    for w in window_list:
        if w.get("kCGWindowNumber") == window_id:
            b = w.get("kCGWindowBounds", {})
            bounds = CG.CGRectMake(
                float(b.get("X", 0)),
                float(b.get("Y", 0)),
                float(b.get("Width", 0)),
                float(b.get("Height", 0)),
            )
            break
    if bounds is None:
        return False

    img = CG.CGWindowListCreateImage(
        bounds,
        CG.kCGWindowListOptionOnScreenOnly,
        CG.kCGNullWindowID,
        CG.kCGWindowImageDefault,
    )
    if img is None:
        return False
    w = CG.CGImageGetWidth(img)
    h = CG.CGImageGetHeight(img)
    if w < 10 or h < 10:
        return False
    return _save_cgimage_to_png(img, output_path)


# Selected capture function (set during init)
_capture_fn = None


def _select_capture_method(window_id, verbose=False):
    """Test all capture methods and select the first that works."""
    global _capture_fn

    methods = [
        (_capture_via_cgimage, "CGImage per-window API"),
        (_capture_via_screencapture, "screencapture command"),
        (_capture_via_region, "CGImage region crop"),
    ]

    for fn, name in methods:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            if fn(window_id, tmp) and os.path.getsize(tmp) > 100:
                _capture_fn = fn
                if verbose:
                    print("  [debug] Using {}".format(name), file=sys.stderr)
                return name
        except Exception:
            pass
        finally:
            _safe_remove(tmp)

    return None


def capture_window(window_id, output_path):
    """Capture a window screenshot using the selected backend."""
    if _capture_fn is not None:
        return _capture_fn(window_id, output_path)
    # No method selected yet — try all
    for fn in [_capture_via_cgimage, _capture_via_screencapture, _capture_via_region]:
        try:
            if fn(window_id, output_path):
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Pixel hashing
# ---------------------------------------------------------------------------

def pixel_hash(path):
    """Hash the decoded pixel data of a PNG, ignoring file-level metadata.

    Decodes the PNG through a CGBitmapContext to get raw RGBA pixels,
    then returns the MD5 hex digest. Returns None on failure.
    """
    provider = CG.CGDataProviderCreateWithFilename(path.encode("utf-8"))
    if provider is None:
        return None
    img = CG.CGImageCreateWithPNGDataProvider(
        provider, None, True, CG.kCGRenderingIntentDefault
    )
    if img is None:
        return None
    w = CG.CGImageGetWidth(img)
    h = CG.CGImageGetHeight(img)
    cs = CG.CGColorSpaceCreateDeviceRGB()
    ctx = CG.CGBitmapContextCreate(
        None, w, h, 8, w * 4, cs, CG.kCGImageAlphaPremultipliedLast
    )
    if ctx is None:
        return None
    CG.CGContextDrawImage(ctx, CG.CGRectMake(0, 0, w, h), img)
    bitmap_img = CG.CGBitmapContextCreateImage(ctx)
    if bitmap_img is None:
        return None
    dp = CG.CGImageGetDataProvider(bitmap_img)
    raw = CG.CGDataProviderCopyData(dp)
    return hashlib.md5(raw).hexdigest()


def capture_and_hash(window_id):
    """Capture to a temp file and return (temp_path, pixel_hash)."""
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    if not capture_window(window_id, tmp):
        _safe_remove(tmp)
        return None, None
    h = pixel_hash(tmp)
    if h is None:
        _safe_remove(tmp)
        return None, None
    return tmp, h


def _safe_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Stabilization: wait until two consecutive captures produce the same hash
# ---------------------------------------------------------------------------

def wait_for_stable_frame(window_id, timeout=6.0, interval=0.4):
    """Poll until two consecutive captures have identical pixel hashes.

    Returns the stable pixel hash, or None on timeout.
    This ensures we don't capture mid-animation frames.
    """
    prev_hash = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        tmp, h = capture_and_hash(window_id)
        if tmp is not None:
            _safe_remove(tmp)
        if h is None:
            time.sleep(interval)
            continue
        if h == prev_hash:
            return h
        prev_hash = h
        time.sleep(interval)
    # Timeout: return the last hash we got (best effort)
    return prev_hash


# ---------------------------------------------------------------------------
# Page turn
# ---------------------------------------------------------------------------

def activate_kindle():
    """Bring Kindle to foreground. No clicking — only activate."""
    subprocess.run(
        ["osascript", "-e", 'tell application id "com.amazon.Lassen" to activate'],
        capture_output=True,
    )
    time.sleep(0.5)


def send_page_turn(direction="rtl", verbose=False):
    """Send a page-forward key to the Kindle process.

    For RTL (right-to-left) books: left arrow (key code 123) = next page.
    For LTR (left-to-right) books: right arrow (key code 124) = next page.

    The key event is sent inside `tell process "Kindle"` to ensure it
    reaches Kindle even if another app momentarily grabs focus.
    """
    key_code = 123 if direction == "rtl" else 124
    script = '''
    tell application "System Events"
        tell process "Kindle"
            set frontmost to true
            delay 0.3
            key code {}
        end tell
    end tell
    '''.format(key_code)
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if verbose and result.returncode != 0:
        print("\n  [debug] page turn osascript failed: {}".format(
            result.stderr.strip()
        ), file=sys.stderr)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Title detection
# ---------------------------------------------------------------------------

def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.strip('. ')
    return name if name else "untitled"


def get_last_opened_asin():
    result = subprocess.run(
        ["defaults", "read", "com.amazon.Lassen", "WasLastOpenedBookId"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        match = re.search(r'[AB]\d[A-Z0-9]{8,}', result.stdout.strip())
        if match:
            return match.group(0)
    return None


def lookup_title_from_homefeed(asin):
    import json
    path = os.path.expanduser(
        "~/Library/Containers/com.amazon.Lassen/Data/Library/Caches/homefeed.json"
    )
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        for card in data.get("cards", []):
            zones = card.get("zones", {})
            bel = zones.get("BOOK_ENTITY_LIST", {})
            if not isinstance(bel, dict):
                continue
            hz = bel.get("homeZone", {})
            if not isinstance(hz, dict):
                continue
            for book in hz.get("bookEntityList", []):
                if book.get("asin") == asin:
                    return book.get("title")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture Kindle book pages as PNG images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --title "My Book"
  %(prog)s --delay 1.0 --max-pages 50
  %(prog)s --verbose                    # show debug output
        """,
    )
    parser.add_argument("--title", help="Book title (auto-detected if omitted)")
    parser.add_argument("--output", default="./captures",
                        help="Output directory (default: ./captures)")
    parser.add_argument("--direction", choices=["rtl", "ltr"], default="rtl",
                        help="Reading direction: rtl for vertical/manga, ltr for horizontal (default: rtl)")
    parser.add_argument("--delay", type=float, default=0.8,
                        help="Base delay after page turn (default: 0.8)")
    parser.add_argument("--start-page", type=int, default=None,
                        help="Starting page number (auto-detected from existing files if omitted)")
    parser.add_argument("--max-pages", type=int, default=5000,
                        help="Max pages to capture (default: 5000)")
    parser.add_argument("--end-retries", type=int, default=5,
                        help="Retries before end-of-book (default: 5)")
    parser.add_argument("--stable-timeout", type=float, default=6.0,
                        help="Timeout for frame stabilization (default: 6.0)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print debug information")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# E2E pre-flight checks
# ---------------------------------------------------------------------------

def preflight_checks(window_id, direction="rtl", verbose=False):
    """Run diagnostics before starting capture. Returns True if OK."""
    ok = True

    # Check 1: Can we capture a screenshot?
    print("  [check] Screenshot capture ... ", end="", flush=True)
    tmp, h = capture_and_hash(window_id)
    if tmp is None or h is None:
        print("FAIL (could not capture or decode)")
        return False
    _safe_remove(tmp)
    print("OK (hash={})".format(h[:12]))

    # Check 2: Is pixel hash stable for a static window?
    print("  [check] Pixel hash stability ... ", end="", flush=True)
    time.sleep(0.3)
    tmp2, h2 = capture_and_hash(window_id)
    if tmp2 is not None:
        _safe_remove(tmp2)
    if h == h2:
        print("OK (stable)")
    else:
        print("WARN (hash changed — cursor blink or animation?)")
        if verbose:
            print("           hash1={} hash2={}".format(h[:12], h2[:12] if h2 else "None"))
        # Not fatal — stabilization loop will handle this

    # Check 3: Does page turn change the content?
    key_name = "left arrow" if direction == "rtl" else "right arrow"
    print("  [check] Page turn ({}) ... ".format(key_name), end="", flush=True)
    before_hash = wait_for_stable_frame(window_id, timeout=3.0)
    send_page_turn(direction=direction, verbose=verbose)
    time.sleep(1.0)
    after_hash = wait_for_stable_frame(window_id, timeout=3.0)
    if before_hash != after_hash:
        print("OK (content changed)")
        # Turn back to restore position
        _send_page_back(direction=direction, verbose=verbose)
        time.sleep(1.0)
    else:
        print("FAIL (content unchanged after page turn)")
        print("\n  The page did not change after pressing the {} key.".format(key_name))
        print("  Possible causes:")
        print("    - No book is open (you're on the library screen)")
        print("    - The reader view does not have keyboard focus")
        print("    - You're on the last page")
        print("    - Wrong reading direction (try --direction ltr)")
        print("\n  Please open a book in Kindle and try again.")
        ok = False

    return ok


def _send_page_back(direction="rtl", verbose=False):
    """Send a page-backward key to go back one page.

    For RTL books: right arrow (key code 124) = previous page.
    For LTR books: left arrow (key code 123) = previous page.
    """
    key_code = 124 if direction == "rtl" else 123
    script = '''
    tell application "System Events"
        tell process "Kindle"
            set frontmost to true
            delay 0.3
            key code {}
        end tell
    end tell
    '''.format(key_code)
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if verbose and result.returncode != 0:
        print("\n  [debug] page back failed: {}".format(result.stderr.strip()),
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------


def detect_last_page(output_dir):
    """Scan output_dir for existing p*.png files and return the max page number.

    Returns (last_page_number, last_file_path) or (0, None) if no files exist.
    """
    pattern = os.path.join(output_dir, "p[0-9]*.png")
    files = _glob.glob(pattern)
    if not files:
        return 0, None
    best_num = 0
    best_path = None
    for f in files:
        basename = os.path.basename(f)
        match = re.match(r'^p(\d+)\.png$', basename)
        if match:
            num = int(match.group(1))
            if num > best_num:
                best_num = num
                best_path = f
    return best_num, best_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    direction = args.direction

    # 0. Screen recording permission check (before anything else)
    print("Checking screen recording permission ... ", end="", flush=True)
    if not check_screen_recording_permission():
        host_app = _detect_host_app()
        print("DENIED\n")
        print("Screen recording permission is required but not granted.")
        print("")
        print("To fix this:")
        print("  1. Open: System Settings > Privacy & Security > Screen Recording")
        print("  2. Enable: \"{}\"".format(host_app))
        print("  3. Restart \"{}\" (required after permission change)".format(host_app))
        print("")
        print("Then run this script again.")
        sys.exit(1)
    print("OK")

    # 1. Process check
    if not is_kindle_running():
        print("Error: Kindle is not running.", file=sys.stderr)
        sys.exit(1)

    # 2. Window detection
    window = find_kindle_window()
    if window is None:
        print("Error: Could not find Kindle window.", file=sys.stderr)
        print("  Make sure Kindle is open and a book is visible on screen.",
              file=sys.stderr)
        sys.exit(1)
    window_id = window["id"]
    print("Found Kindle window: id={}, size={}x{}".format(
        window_id, window["width"], window["height"]
    ))
    print("Reading direction: {} ({})".format(
        direction, "right-to-left" if direction == "rtl" else "left-to-right"
    ))

    # 2.5. Select capture method
    print("Selecting capture method ... ", end="", flush=True)
    method_name = _select_capture_method(window_id, verbose=args.verbose)
    if method_name:
        print("OK ({})".format(method_name))
    else:
        print("FAIL")
        print("\nCould not capture the Kindle window with any method.", file=sys.stderr)
        print("Possible causes:", file=sys.stderr)
        print("  - Kindle window is minimized or fully behind other windows", file=sys.stderr)
        print("  - Screen recording permission was just granted (restart the terminal)", file=sys.stderr)
        sys.exit(1)

    # 3. Title
    title = None
    if args.title:
        title = args.title
    else:
        wname = window.get("name", "")
        if wname and wname != "Kindle":
            title = wname
        else:
            asin = get_last_opened_asin()
            if asin:
                title = lookup_title_from_homefeed(asin)
                if title:
                    print("Auto-detected: '{}'".format(title))
                else:
                    print("ASIN: {} (title not in cache)".format(asin))
        if not title:
            try:
                title = input("Enter book title: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.", file=sys.stderr)
                sys.exit(1)
            if not title:
                sys.exit(1)

    output_dir = os.path.join(args.output, sanitize_filename(title))
    os.makedirs(output_dir, exist_ok=True)
    print("Output: {}".format(os.path.abspath(output_dir)))

    # 4. Resume detection
    resuming = False
    if args.start_page is not None:
        page_num = args.start_page
    else:
        last_page, last_file = detect_last_page(output_dir)
        if last_page > 0:
            page_num = last_page
            resuming = True
            print("Resuming: found {} existing pages (last: p{:04d}.png)".format(
                last_page, last_page))
        else:
            page_num = 1

    # 5. Activate Kindle (no click)
    activate_kindle()

    # 6. Pre-flight checks
    print("\nRunning pre-flight checks:")
    if not preflight_checks(window_id, direction=direction, verbose=args.verbose):
        sys.exit(1)
    print()

    # Need to re-stabilize after preflight (which turns a page back)
    time.sleep(0.5)

    # Refresh window ID after preflight
    window = find_kindle_window()
    if window is None:
        print("Error: Kindle window lost after preflight.", file=sys.stderr)
        sys.exit(1)
    window_id = window["id"]

    # 7. Set up initial state
    captured = 0
    end_retry_count = 0

    # Get stable hash of current page
    current_hash = wait_for_stable_frame(window_id, timeout=args.stable_timeout)
    if current_hash is None:
        print("Error: Could not get stable screenshot.", file=sys.stderr)
        sys.exit(1)

    if resuming:
        # When resuming, use the last saved file's hash as the baseline.
        # The user has manually navigated Kindle to the correct page.
        last_file = os.path.join(output_dir, "p{:04d}.png".format(page_num))
        if os.path.exists(last_file):
            saved_hash = pixel_hash(last_file)
            if saved_hash is not None:
                current_hash = saved_hash
        print("Starting capture from page {} ... (Ctrl+C to stop)\n".format(
            page_num + 1))
    else:
        # Save first page
        print("Starting capture... (Ctrl+C to stop)\n")
        first_path = os.path.join(output_dir, "p{:04d}.png".format(page_num))
        if not capture_window(window_id, first_path):
            print("Error: Failed to save first page.", file=sys.stderr)
            sys.exit(1)
        saved_hash = pixel_hash(first_path)
        if saved_hash is not None:
            current_hash = saved_hash
        captured = 1
        sys.stdout.write("\rCaptured: {} pages".format(captured))
        sys.stdout.flush()

    # 8. Main capture loop
    try:
        while captured < args.max_pages:
            # Check process
            if not is_kindle_running():
                print("\nKindle process exited.", file=sys.stderr)
                break

            # Refresh window ID
            fresh = find_kindle_window()
            if fresh is None:
                print("\nKindle window disappeared.", file=sys.stderr)
                break
            window_id = fresh["id"]

            prev_hash = current_hash

            # --- Turn page ---
            if not send_page_turn(direction=direction, verbose=args.verbose):
                if args.verbose:
                    print("\n  [debug] page turn command failed", file=sys.stderr)
                # Try re-activating
                activate_kindle()
                send_page_turn(direction=direction, verbose=args.verbose)

            # Base delay for rendering to begin
            time.sleep(args.delay)

            # --- Wait for stable frame ---
            stable_hash = wait_for_stable_frame(
                window_id, timeout=args.stable_timeout
            )

            if stable_hash is None:
                if args.verbose:
                    print("\n  [debug] stabilization timeout", file=sys.stderr)
                end_retry_count += 1
                if end_retry_count >= args.end_retries:
                    print("\nEnd of book (stabilization failed {} times).".format(
                        args.end_retries))
                    break
                activate_kindle()
                continue

            # --- Compare with previous page ---
            if stable_hash == prev_hash:
                # Page didn't change
                end_retry_count += 1
                if end_retry_count >= args.end_retries:
                    print("\nEnd of book (page unchanged after {} retries).".format(
                        args.end_retries))
                    break
                if args.verbose:
                    sys.stdout.write(
                        "\r  [retry {}/{}] Page unchanged, retrying...    ".format(
                            end_retry_count, args.end_retries))
                    sys.stdout.flush()
                # Re-activate and retry
                activate_kindle()
                time.sleep(0.3)
                continue

            # --- Page changed: save ---
            end_retry_count = 0
            page_num += 1
            filepath = os.path.join(output_dir, "p{:04d}.png".format(page_num))

            if not capture_window(window_id, filepath):
                print("\nError: Failed to save p{:04d}.png".format(page_num),
                      file=sys.stderr)
                break

            # Use the hash of the actually saved file
            final_hash = pixel_hash(filepath)
            current_hash = final_hash if final_hash is not None else stable_hash

            captured += 1
            sys.stdout.write("\rCaptured: {} pages (p{:04d}.png)".format(
                captured, page_num))
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    print("\nDone. {} pages saved in: {}".format(
        captured, os.path.abspath(output_dir)))


if __name__ == "__main__":
    main()
