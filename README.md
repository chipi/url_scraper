# URL Scraper (single-file crawler)

A small, single-file Python crawler that discovers and lists URLs starting from a source page.

## Requirements

- Python 3.8+
- No external dependencies (uses Python standard library)

## File

- `url_scraper.py` — CLI entry point and crawler implementation

## Usage

```bash
python3 url_scraper.py <url> [options]
```

Examples:

```bash
# Basic crawl (same-domain, depth 3, up to 200 pages)
python3 url_scraper.py https://example.com

# Increase depth and pages, save to a file
python3 url_scraper.py https://example.com --max-depth 4 --max-pages 500 -o urls.txt

# Crawl across all domains (not just the start domain)
python3 url_scraper.py https://example.com --all-domains

# Ignore robots.txt (use responsibly)
python3 url_scraper.py https://example.com --no-robots

# Add delay between requests (politeness)
python3 url_scraper.py https://example.com --delay-ms 250

# Skip paths like /videos and /assets during crawl
python3 url_scraper.py https://example.com --ignore-path /videos --ignore-path /assets

# Comma-separated ignore list in a single flag
python3 url_scraper.py https://example.com --ignore-path /videos,/assets,/downloads

# Download cleaned main-content as text files into a directory (one file per URL)
python3 url_scraper.py https://example.com --download-dir ./pages
```

## Options

- `--max-pages` (int): Maximum pages to fetch (default: 200)
- `--max-depth` (int): Maximum crawl depth from the start URL (default: 3)
- `--all-domains`: Do not restrict to the start URL's domain
- `--no-robots`: Ignore robots.txt (use responsibly)
- `--user-agent` (str): User-Agent header to use
- `--timeout` (int): Request timeout in seconds (default: 15)
- `--delay-ms` (int): Politeness delay between requests in milliseconds
- `--ignore-path` (repeatable or comma-separated): Path prefixes to skip (e.g., `/videos`, `/assets`)
- `-o, --output` (path): Optional output file to write URLs to (one per line)
- `--download-dir` (path): Optional directory to save cleaned main-content as `.txt` files (one per URL)

## Notes

- By default, the crawler respects `robots.txt` and stays within the start domain.
- URLs are normalized (fragment removed, default ports stripped) and printed one per line.
- If `--download-dir` is provided, each crawled page is fetched again and saved as a plain text file containing:
  - Title (or URL if no title), URL, and underline header
  - Cleaned main content (heuristic: prefers `<main>`, `<article>`, `role="main"`; excludes nav/header/footer/aside/script/style)
  - A "Reference links" section listing in-content links normalized to absolute URLs

