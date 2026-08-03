#!/usr/bin/env python3
"""Check external (http/https) URLs referenced in maintained docs.

The in-repo audit (doc/development/docs_audit.py) validates *relative* links
only. This script extends coverage to external URLs so moved/retired resources
can be caught before they rot.

Run from repository root:

  uv run python scripts/check_external_links.py            # report only
  uv run python scripts/check_external_links.py --strict   # exit 1 on dead links
  uv run python scripts/check_external_links.py --concurrency 8 --timeout 15

Design notes:

- **Informational by default.** External sites rate-limit or bot-block
  automated clients (crates.io, paperswithcode, Wikipedia are common), so a
  non-2xx response is *not* proof a link is dead. Use ``--strict`` only when
  you intend to triage every finding; the check is intentionally NOT wired
  into CI.
- Local/template targets (localhost, 127.0.0.1, example.com, ``{host}``
  templates, ``server:port``) are skipped by design — they are configuration
  examples, not links.
- Findings are grouped by status code; 4xx/000 are the interesting ones.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAINTAINED_ROOTS = [
    ROOT / ".agent_rules",
    ROOT / ".github",
    ROOT / "doc",
    ROOT / "input",
    ROOT / "manuscript",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "AGENTS.md",
    ROOT / "AGENTS_TEMPLATE.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CLAUDE.md",
    ROOT / "CODE_OF_CONDUCT.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "DOCS.md",
    ROOT / "README.md",
    ROOT / "SECURITY.md",
    ROOT / "SETUP_GUIDE.md",
    ROOT / "SKILL.md",
    ROOT / "SPEC.md",
    ROOT / "SUPPORT.md",
    ROOT / "TO-DO.md",
]

SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "archive",
    "build",
    "dist",
    "node_modules",
    "output",
    ".benchmarks",
    ".aii",
}

URL_RE = re.compile(r"https?://[^\s)\]>\"']+")

# Hosts/paths that are configuration examples or local services, never links.
SKIP_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "example.com",
    "www.example.com",
    "docs.example.com",
    "api.example.com",
    "mcp-server.example.com",
    "yourdomain.com",
    "gnn_backends",
    "server",
    "colbert-server",
    "<server-ip",
}
TEMPLATE_MARKERS = ("{", "}", "<server-ip", "server:port", "yourdomain")


def _is_generated(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return True
    parts = rel.parts
    if any(p in SKIP_PARTS for p in parts):
        return True
    if any(p.startswith("activeinference_outputs_") for p in parts):
        return True
    if any(p.endswith("_outputs") or "_outputs_" in p for p in parts):
        return True
    return "pomdp_gridworld_outputs" in parts


def _should_skip_url(url: str) -> bool:
    host = url.split("/", 3)[2].lower() if "//" in url else ""
    host = host.split(":")[0].split("@")[-1]
    if host in SKIP_HOSTS:
        return True
    if host == "example.com" or host.endswith(".example.com"):
        return True
    if any(m in url for m in TEMPLATE_MARKERS):
        return True
    return False


def collect() -> dict[str, list[str]]:
    urls: dict[str, list[str]] = {}
    for root in MAINTAINED_ROOTS:
        if not root.exists() or _is_generated(root):
            continue
        candidates = (
            [root]
            if root.is_file()
            else sorted(p for p in root.rglob("*") if p.is_file())
        )
        for path in candidates:
            if path.suffix not in {".md", ".txt", ".yaml", ".yml", ".toml", ".cfg"}:
                continue
            if _is_generated(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for m in URL_RE.finditer(text):
                url = m.group(0).rstrip(".,;:!?`")
                # Re-balance a trailing paren the character class cut off
                # (e.g. https://learn.microsoft.com/.../ms256108(v=vs.85)).
                if url.count("(") > url.count(")"):
                    url += ")"
                if _should_skip_url(url):
                    continue
                urls.setdefault(url, []).append(str(path.relative_to(ROOT)))
    return urls


def check(url: str, timeout: int) -> tuple[str, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return str(resp.status), url
    except urllib.error.HTTPError as exc:
        return str(exc.code), url
    except Exception as exc:  # noqa: BLE001
        return f"ERR:{type(exc).__name__}", url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="Exit 1 when dead links are found."
    )
    parser.add_argument(
        "--concurrency", type=int, default=16, help="Parallel checks (default: 16)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Per-URL timeout in seconds (default: 20).",
    )
    parser.add_argument(
        "--min-status",
        type=int,
        default=400,
        help="Report statuses >= this value (default: 400).",
    )
    args = parser.parse_args()

    urls = collect()
    print(
        f"check_external_links: {len(urls)} unique external URLs across maintained docs",
        file=sys.stderr,
    )

    results: list[tuple[str, str]] = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(check, url, args.timeout) for url in sorted(urls)]
        for fut in cf.as_completed(futures):
            results.append(fut.result())

    by_code: dict[str, list[str]] = defaultdict(list)
    for code, url in results:
        by_code[code].append(url)

    counts = Counter(code for code, _ in results)
    print("summary:", dict(sorted(counts.items())))
    print()

    interesting = {
        c
        for c in by_code
        if c.startswith("ERR") or (c.isdigit() and int(c) >= args.min_status)
    }
    dead = 0
    for code in sorted(
        interesting, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0)
    ):
        for url in sorted(by_code[code]):
            refs = ", ".join(sorted(set(urls[url]))[:4])
            extra = (
                f" (+{len(set(urls[url])) - 4} more)" if len(set(urls[url])) > 4 else ""
            )
            print(f"  {code} {url}")
            print(f"      <- {refs}{extra}")
            dead += 1
    if not dead:
        print("no findings above the configured threshold")

    if args.strict and dead:
        print(f"\ncheck_external_links: {dead} finding(s); exiting 1 (--strict).")
        return 1
    print(
        "\n(non-strict: exit 0; use --strict to fail — triage bot-blocked hosts first)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
