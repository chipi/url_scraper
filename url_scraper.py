#!/usr/bin/env python3

import argparse
import collections
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


def crawl(config: CrawlConfig) -> List[str]:
    start = normalize_url(config.start_url, config.start_url)
    if not start:
        return []

    visited: Set[str] = set()
    discovered: Set[str] = set()
    queue: collections.deque[Tuple[str, int]] = collections.deque()
    queue.append((start, 0))

    rp = build_robot_parser(start, config.user_agent, config.timeout) if config.respect_robots else None

    while queue and len(visited) < config.max_pages:
        current, depth = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

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
            if link not in visited:
                queue.append((link, depth + 1))

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
    )

    urls = crawl(cfg)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                for u in urls:
                    f.write(u + "\n")
        except Exception as e:
            print(f"Failed to write output file: {e}", file=sys.stderr)
            # Still print to stdout

    for u in urls:
        print(u)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


