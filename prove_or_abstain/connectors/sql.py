"""connectors/sql.py — pull a long-panel DataFrame from a SQL database
(Postgres, MySQL, SQLite, or anything else SQLAlchemy has a driver for)
via a user-supplied query, instead of a CSV upload.

Trust model: the caller already holds the DSN/credentials for their own
database — this module grants no access it wouldn't otherwise have. The
one guard it adds is scoping each call to a single read-only statement, so
a pasted-in query can't accidentally (or maliciously) mutate the database
through this endpoint: only SELECT / WITH ... SELECT is allowed, and only
one statement per call.

This is a safety rail against accidents, not a SQL parser — it does not
attempt to catch every way a database driver might execute multiple
statements (e.g. driver-level multi-statement extensions). Treat the
connection as read-only at the account level (a reporting replica or a
role with SELECT-only grants) for real deployments.
"""
from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import create_engine

_READ_ONLY_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
# Single/double-quoted literals, handling standard SQL '' / "" escaping —
# stripped before the ';' count so a legitimate WHERE note = 'a; b' isn't
# mistaken for a second statement. Still not a real SQL parser (see below).
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")


class SqlQueryError(ValueError):
    """Raised for a rejected or failing query — never for a database
    connection the caller shouldn't have (that's on their DSN/grants)."""


def _guard_single_select(query: str) -> None:
    if not _READ_ONLY_RE.match(query):
        raise SqlQueryError("only a SELECT (or WITH ... SELECT) query is allowed")
    without_literals = _STRING_LITERAL_RE.sub("", query.strip().rstrip(";"))
    if ";" in without_literals:
        raise SqlQueryError("only a single statement is allowed (remove the ';')")


def fetch_panel(dsn: str, query: str) -> pd.DataFrame:
    """Run one read-only query against `dsn` and return the result rows
    as a DataFrame. The query is expected to already project onto the
    long-panel shape [metric, <dims...>, n, c] — this function does no
    reshaping, only the transport."""
    _guard_single_select(query)
    engine = create_engine(dsn)
    try:
        return pd.read_sql_query(query, engine)
    except SqlQueryError:
        raise
    except Exception as exc:
        raise SqlQueryError(f"query failed: {exc}") from exc
    finally:
        engine.dispose()
