"""connectors/gsheets.py — pull a long-panel DataFrame from a Google Sheet.

Accepts any of the usual sheet URLs (a share link, an "edit#gid=" link, or
an already-built CSV export link) and normalizes it to the sheet's CSV
export endpoint, then downloads and parses it exactly like an uploaded CSV.

Trust model: the connector only ever calls docs.google.com — a URL pointing
anywhere else is rejected before any request is made, so this can't be
used as an open URL fetcher / SSRF proxy. The sheet itself must already be
shared as "anyone with the link" (or published to the web); this module
does not authenticate and has no Google API credentials.
"""
from __future__ import annotations

import io
import re
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

_ALLOWED_HOST = "docs.google.com"
_ID_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")


class SheetError(ValueError):
    """Raised for a rejected/unreachable/malformed sheet — never for a
    resource outside docs.google.com (see _to_csv_url)."""


def _to_csv_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.netloc != _ALLOWED_HOST:
        raise SheetError(f"only a {_ALLOWED_HOST} spreadsheet URL is accepted")

    if parsed.path.endswith("/export") and "format=csv" in parsed.query:
        return url  # already a CSV export link — use as-is

    m = _ID_RE.search(parsed.path)
    if not m:
        raise SheetError("could not find a spreadsheet id in the URL")
    sheet_id = m.group(1)

    gid = "0"
    qs = parse_qs(parsed.query)
    if "gid" in qs:
        gid = qs["gid"][0]
    elif parsed.fragment.startswith("gid="):
        gid = parsed.fragment.split("=", 1)[1]

    return f"https://{_ALLOWED_HOST}/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def fetch_panel(url: str) -> pd.DataFrame:
    """Download one Google Sheet (or one tab, via gid) as a long-panel
    DataFrame [metric, <dims...>, n, c]. Raises SheetError on anything that
    isn't a readable, publicly-viewable sheet."""
    csv_url = _to_csv_url(url)
    try:
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SheetError(f"could not fetch the sheet: {exc}") from exc

    if resp.text.lstrip().startswith("<"):
        # a private/unshared sheet serves an HTML sign-in page, not a CSV
        raise SheetError("the sheet is not publicly viewable — share it as "
                         "'anyone with the link' or publish it to the web")
    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        raise SheetError(f"could not parse the sheet as CSV: {exc}") from exc
    if df.empty:
        raise SheetError("the sheet (or tab) has no rows")
    return df
