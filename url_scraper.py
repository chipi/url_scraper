#!/usr/bin/env python3

import argparse
import collections
import os
import re
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable, List, Optional, Set, Tuple
from urllib import robotparser
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse, urldefrag
from urllib.request import Request, build_opener, HTTPSHandler, HTTPHandler


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0 Safari/537.36"
)


class LinkExtractor(HTMLParser):
    """Extracts href links from anchor-like tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag not in ("a", "area", "link"):
            return
        href = None
        for (attr, val) in attrs:
            if attr.lower() == "href":
                href = val
                break
        if href:
            self.links.append(href)


def normalize_url(base_url: str, raw_link: str) -> Optional[str]:
    """Resolve and canonicalize a URL. Returns None if scheme unsupported."""
    # Resolve relative
    resolved = urljoin(base_url, raw_link)

    # Remove fragment
    resolved, _ = urldefrag(resolved)

    parts = urlparse(resolved)
    if parts.scheme not in ("http", "https"):
        return None

    # Normalize: lowercase scheme/host; strip default ports; remove redundant path segments
    netloc = parts.netloc
    hostname = parts.hostname or ""
    port = parts.port
    if port is None:
        netloc = hostname
    else:
        # Keep non-default ports
        is_default = (parts.scheme == "http" and port == 80) or (parts.scheme == "https" and port == 443)
        netloc = hostname if is_default else f"{hostname}:{port}"

    normalized = urlunparse(
        (
            parts.scheme.lower(),
            netloc.lower(),
            parts.path or "/",
            "",
            parts.query,
            "",
        )
    )
    return normalized


def is_same_domain(seed_url: str, candidate_url: str) -> bool:
    a = urlparse(seed_url)
    b = urlparse(candidate_url)
    return a.hostname == b.hostname


def fetch_html(url: str, user_agent: str, timeout: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch URL and return (html_text, final_url) if content is HTML, else (None, final_url).
    Uses urllib to avoid third-party dependencies.
    """
    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"}
    req = Request(url, headers=headers)
    opener = build_opener(HTTPHandler(), HTTPSHandler())
    try:
        with opener.open(req, timeout=timeout) as resp:
            final_url = resp.geturl()
            ctype = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
                return None, final_url
            charset = "utf-8"
            if "charset=" in ctype:
                try:
                    charset = ctype.split("charset=", 1)[1].split(";")[0].strip()
                except Exception:
                    pass
            try:
                body_bytes = resp.read()
            except Exception:
                return None, final_url
            try:
                html = body_bytes.decode(charset, errors="replace")
            except Exception:
                html = body_bytes.decode("utf-8", errors="replace")
            return html, final_url
    except (HTTPError, URLError):
        return None, url


def extract_links(base_url: str, html: str) -> List[str]:
    parser = LinkExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # If parsing fails, return what we have
        pass
    normalized: List[str] = []
    for raw in parser.links:
        norm = normalize_url(base_url, raw)
        if norm:
            normalized.append(norm)
    return normalized


class MainContentExtractor(HTMLParser):
    """
    Heuristic main-content text extractor using only stdlib.
    - Prefers content within <main>, <article>, or role="main".
    - Excludes typical chrome: nav, header, footer, aside, menu, form, script, style, noscript.
    - Captures text and in-content links; simple paragraph handling.
    """

    EXCLUDE_TAGS = {"script", "style", "noscript", "template", "svg"}
    CHROME_TAGS = {"nav", "header", "footer", "aside", "menu", "form"}
    BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "main",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: List[str] = []
        self._in_main_like: int = 0
        self._suppress: int = 0
        self._buffer: List[str] = []
        self._paragraph_open: bool = False
        self._links: List[str] = []
        self._title: List[str] = []
        self._in_title: bool = False
        # Attribute-based exclusion patterns (breadcrumbs, share, back, meta, related)
        self._exclude_attr_regex = re.compile(
            r"\b(breadcrumbs?|crumb|share|social|sns|back(-to)?|post-(meta|nav)|byline|related|tags|tag-list|follow-us|share-tools)\b",
            re.IGNORECASE,
        )

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        self._stack.append(tag_lower)

        if tag_lower == "title":
            self._in_title = True

        if tag_lower in self.EXCLUDE_TAGS:
            self._suppress += 1
            return

        # Detect main-like region
        if tag_lower in ("main", "article"):
            self._in_main_like += 1
        else:
            # role="main"
            for (attr, val) in attrs:
                if attr.lower() == "role" and (val or "").lower() == "main":
                    self._in_main_like += 1
                    break

        # Exclude chrome sections fully if outside an explicit main-like already
        if self._in_main_like == 0 and tag_lower in self.CHROME_TAGS:
            self._suppress += 1

        # Inside main-like, suppress typical non-content blocks by class/id/aria-label/role
        if self._in_main_like > 0 and self._suppress == 0:
            if self._attrs_match_decorative(attrs):
                self._suppress += 1

        # Capture links when in main-like and not suppressed
        if self._in_main_like > 0 and self._suppress == 0 and tag_lower == "a":
            href_val: Optional[str] = None
            for (attr, val) in attrs:
                if attr.lower() == "href":
                    href_val = val
                    break
            if href_val:
                self._links.append(href_val)

        # Paragraph-like separation
        if tag_lower in self.BLOCK_TAGS and self._in_main_like > 0 and self._suppress == 0:
            if self._paragraph_open:
                self._buffer.append("\n\n")
            self._paragraph_open = True

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        # Title handling
        if tag_lower == "title":
            self._in_title = False

        if tag_lower in self.EXCLUDE_TAGS and self._suppress > 0:
            self._suppress -= 1

        if tag_lower in ("main", "article") and self._in_main_like > 0:
            self._in_main_like -= 1

        if self._in_main_like == 0 and tag_lower in self.CHROME_TAGS and self._suppress > 0:
            self._suppress -= 1

        # Close suppression for decorative blocks matched by attrs
        if self._in_main_like >= 0 and self._suppress > 0 and self._stack:
            # Best-effort: when closing current tag, allow suppression to unwind
            # only if this end tag corresponds to a block-level likely suppressed section
            if tag_lower in {"div", "section", "ul", "ol", "nav", "header", "footer", "aside"}:
                self._suppress -= 1

        if tag_lower in self.BLOCK_TAGS and self._paragraph_open:
            # End of paragraph-like block
            self._paragraph_open = False

        if self._stack and self._stack[-1] == tag_lower:
            self._stack.pop()

    def _attrs_match_decorative(self, attrs: List[Tuple[str, Optional[str]]]) -> bool:
        for (attr, val) in attrs:
            if val is None:
                continue
            name = attr.lower()
            value = str(val)
            if name in ("class", "id", "aria-label", "role", "name"):
                if self._exclude_attr_regex.search(value):
                    return True
        return False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        if self._in_main_like == 0 or self._suppress > 0:
            return
        text = data.strip()
        if not text:
            return
        # Collapse internal whitespace
        text = re.sub(r"\s+", " ", text)
        self._buffer.append(text + " ")

    def get_title(self) -> str:
        title = re.sub(r"\s+", " ", "".join(self._title).strip())
        return title

    def get_text(self) -> str:
        # Normalize whitespace; ensure paragraphs are separated
        text = "".join(self._buffer)
        # Clean up excessive spaces around newlines
        text = re.sub(r"\s*\n\s*", "\n", text)
        # Collapse multiple newlines to max two
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Final trim
        text = text.strip()
        return text

    def get_links(self) -> List[str]:
        return list(self._links)


def extract_main_content_and_links(base_url: str, html: str) -> Tuple[str, str, List[str]]:
    """
    Returns (title, text, reference_links).
    Links are normalized and de-duplicated, relative to base_url.
    """
    parser = MainContentExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    title = parser.get_title()
    text = parser.get_text()
    text = clean_decorative_blocks(text)

    raw_links = parser.get_links()
    refs: List[str] = []
    seen: Set[str] = set()
    for raw in raw_links:
        norm = normalize_url(base_url, raw)
        if norm and norm not in seen:
            seen.add(norm)
            refs.append(norm)

    return title, text, refs


def _is_decorative_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    # Common back/share phrases
    if re.search(r"^(←|<−|<–|<-)?\s*(back|go back|return|previous|home)\b", s, re.IGNORECASE):
        return True
    if re.search(r"\b(share|share this|follow us|connect with us)\b", s, re.IGNORECASE):
        return True
    # Lines mostly consisting of social network names
    if re.search(r"\b(twitter|x|facebook|fb|linkedin|reddit|email|whatsapp|wechat|telegram|pinterest)\b", s, re.IGNORECASE):
        # Short lines with only social names are decorative
        if len(s) <= 64:
            return True
    return False


def clean_decorative_blocks(text: str) -> str:
    if not text.strip():
        return text
    lines = text.split("\n")
    # Trim leading decorative lines
    start = 0
    while start < len(lines) and _is_decorative_line(lines[start]):
        start += 1
    # Trim trailing decorative lines
    end = len(lines) - 1
    while end >= start and _is_decorative_line(lines[end]):
        end -= 1
    cleaned = lines[start : end + 1]
    # Also drop isolated short decorative blocks inside: collapse sequences
    out: List[str] = []
    i = 0
    while i < len(cleaned):
        if _is_decorative_line(cleaned[i]):
            # Collapse multiple decorative lines into a single blank between content blocks
            while i < len(cleaned) and _is_decorative_line(cleaned[i]):
                i += 1
            if out and i < len(cleaned):
                out.append("")
            continue
        out.append(cleaned[i])
        i += 1
    # Remove excessive blank lines
    joined = "\n".join(out)
    joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
    return joined


def sanitize_filename_from_url(url: str, max_len: int = 180) -> str:
    parts = urlparse(url)
    host = (parts.hostname or "site").lower()
    path = parts.path or "/"
    if path.endswith("/"):
        path = path[:-1]
    if not path:
        path = "/"
    # Replace separators with dashes; keep alnum and a few safe chars
    raw = host + ("-" + path.strip("/").replace("/", "-") if path != "/" else "")
    raw = raw if raw else host
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", raw)
    raw = re.sub(r"-+", "-", raw).strip("-._")
    if not raw:
        raw = host
    # Append short hash from full URL to avoid collisions when needed
    suffix = hex(abs(hash(url)) & 0xFFFF)[2:]
    name = raw[: max(1, max_len - len(suffix) - 1)] + "-" + suffix
    return name + ".txt"


def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def build_text_document(url: str, title: str, body_text: str, refs: List[str]) -> str:
    header_lines: List[str] = []
    title_line = title.strip() if title.strip() else url
    header_lines.append(title_line)
    header_lines.append(url)
    header_lines.append("".rjust(len(url), "="))
    header = "\n".join(header_lines)

    sections = [header]
    if body_text.strip():
        sections.append("")
        sections.append(body_text.strip())
    if refs:
        sections.append("")
        sections.append("Reference links:")
        sections.append("-----------------")
        sections.extend(refs)
    return "\n".join(sections).rstrip() + "\n"


@dataclass
class CrawlConfig:
    start_url: str
    max_pages: int = 200
    max_depth: int = 3
    same_domain_only: bool = True
    respect_robots: bool = True
    user_agent: str = DEFAULT_USER_AGENT
    timeout: int = 15
    delay_ms: int = 0
    ignore_paths: List[str] = None  # path prefixes like /videos, /assets


def build_robot_parser(start_url: str, user_agent: str, timeout: int) -> Optional[robotparser.RobotFileParser]:
    parts = urlparse(start_url)
    robots_url = urlunparse((parts.scheme, parts.netloc, "/robots.txt", "", "", ""))
    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        # robotparser uses urllib under the hood
        rp.read()
        rp.useragent = user_agent
        return rp
    except Exception:
        return None


def _normalize_ignore_paths(ignore_paths: Optional[List[str]]) -> List[str]:
    if not ignore_paths:
        return []
    normalized: List[str] = []
    for raw in ignore_paths:
        if not raw:
            continue
        for part in [p.strip() for p in str(raw).split(",")]:
            if not part:
                continue
            p = part if part.startswith("/") else "/" + part
            # Remove trailing slashes for consistent prefix comparison (except root)
            if len(p) > 1:
                p = p.rstrip("/")
            normalized.append(p)
    return normalized


def crawl(config: CrawlConfig) -> List[str]:
    start = normalize_url(config.start_url, config.start_url)
    if not start:
        return []

    visited: Set[str] = set()
    discovered: Set[str] = set()
    queue: collections.deque[Tuple[str, int]] = collections.deque()
    queue.append((start, 0))

    ignored_prefixes = _normalize_ignore_paths(config.ignore_paths)

    rp = build_robot_parser(start, config.user_agent, config.timeout) if config.respect_robots else None

    while queue and len(visited) < config.max_pages:
        current, depth = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        try:
            sys.stderr.write(f"[{len(visited)}/{config.max_pages}] depth={depth} fetching {current}\n")
            sys.stderr.flush()
        except Exception:
            pass

        if config.respect_robots and rp is not None:
            try:
                if not rp.can_fetch(config.user_agent, current):
                    continue
            except Exception:
                # If robots parsing fails for this URL, proceed conservatively by skipping
                continue

        html, final_url = fetch_html(current, config.user_agent, config.timeout)
        if html is None:
            continue

        final_norm = normalize_url(final_url, final_url) or current
        discovered.add(final_norm)

        if depth >= config.max_depth:
            continue

        links = extract_links(final_norm, html)
        for link in links:
            if config.same_domain_only and not is_same_domain(start, link):
                continue
            if ignored_prefixes:
                link_path = urlparse(link).path or "/"
                # Normalize path similar to ignore prefixes: remove trailing slash except root
                if len(link_path) > 1:
                    link_path = link_path.rstrip("/")
                if any(link_path.startswith(prefix) for prefix in ignored_prefixes):
                    continue
            if link not in visited:
                queue.append((link, depth + 1))

        try:
            sys.stderr.write(
                f"    + links found: {len(links)} | discovered: {len(discovered)} | queue: {len(queue)}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass

        if config.delay_ms > 0:
            time.sleep(config.delay_ms / 1000.0)

    return sorted(discovered)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl a website starting from a source URL and list discovered page URLs.",
    )
    parser.add_argument("url", help="Source URL to start crawling from (http/https)")
    parser.add_argument("--max-pages", type=int, default=200, help="Maximum pages to fetch (default: 200)")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum crawl depth from the start URL (default: 3)")
    parser.add_argument(
        "--all-domains", action="store_true", help="Do not restrict to the start URL's domain"
    )
    parser.add_argument(
        "--no-robots", action="store_true", help="Ignore robots.txt (use responsibly)"
    )
    parser.add_argument(
        "--user-agent", default=DEFAULT_USER_AGENT, help="User-Agent header to use"
    )
    parser.add_argument("--timeout", type=int, default=15, help="Request timeout in seconds (default: 15)")
    parser.add_argument("--delay-ms", type=int, default=0, help="Politeness delay between requests in ms")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output file to write URLs to (one per line)",
    )
    parser.add_argument(
        "--ignore-path",
        action="append",
        default=[],
        help="Path prefixes to skip (repeatable or comma-separated), e.g. /videos,/assets",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Optional directory to save cleaned main-content as .txt files (one per URL)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    cfg = CrawlConfig(
        start_url=args.url,
        max_pages=max(1, args.max_pages),
        max_depth=max(0, args.max_depth),
        same_domain_only=not args.all_domains,
        respect_robots=not args.no_robots,
        user_agent=args.user_agent,
        timeout=max(1, args.timeout),
        delay_ms=max(0, args.delay_ms),
        ignore_paths=args.ignore_path,
    )

    try:
        sys.stderr.write(
            "Starting crawl\n"
            f"  source: {cfg.start_url}\n"
            f"  max_pages: {cfg.max_pages}, max_depth: {cfg.max_depth}\n"
            f"  same_domain_only: {cfg.same_domain_only}, respect_robots: {cfg.respect_robots}\n"
        )
        sys.stderr.flush()
    except Exception:
        pass

    urls = crawl(cfg)

    try:
        sys.stderr.write(
            f"Finished. pages_fetched: {len(urls)} (unique URLs listed).\n"
        )
        sys.stderr.flush()
    except Exception:
        pass

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                for u in urls:
                    f.write(u + "\n")
        except Exception as e:
            print(f"Failed to write output file: {e}", file=sys.stderr)
            # Still print to stdout

    # Optional: download cleaned content for each URL
    if args.download_dir:
        ensure_dir(args.download_dir)
        try:
            sys.stderr.write(f"Downloading cleaned content into: {args.download_dir}\n")
            sys.stderr.flush()
        except Exception:
            pass

        saved = 0
        for u in urls:
            html, final_url = fetch_html(u, cfg.user_agent, cfg.timeout)
            if html is None:
                continue
            base = normalize_url(final_url or u, final_url or u) or u
            title, text, refs = extract_main_content_and_links(base, html)
            # Build document and write
            doc = build_text_document(base, title, text, refs)
            filename = sanitize_filename_from_url(base)
            file_path = os.path.join(args.download_dir, filename)
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(doc)
                saved += 1
                try:
                    sys.stderr.write(f"  saved: {file_path}\n")
                    sys.stderr.flush()
                except Exception:
                    pass
            except Exception:
                continue

        try:
            sys.stderr.write(f"Saved {saved} files.\n")
            sys.stderr.flush()
        except Exception:
            pass

    for u in urls:
        print(u)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


