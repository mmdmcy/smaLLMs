"""
Benchmark rows from Hugging Face over plain HTTPS, on the standard library.

Rows come from Hugging Face's dataset viewer API
(https://datasets-server.huggingface.co), at most 100 rows per request, paged
lazily so a benchmark that needs 10 rows asks for one page. ``/filter`` serves
the long-context benchmarks that select rows by a column. A dataset the viewer
does not serve has its own loader in benchmarks.py.

A token is sent as ``Authorization: Bearer`` only when one is configured
(HF_TOKEN, HUGGINGFACE_HUB_TOKEN, HF_KEY or the repository's .env; see
src/utils/env.py).
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Optional

from src.utils import http_client
from src.utils.env import huggingface_token

LOGGER = logging.getLogger(__name__)

VIEWER_URL = "https://datasets-server.huggingface.co"
HUB_URL = "https://huggingface.co"
PAGE_SIZE = 100
REQUEST_TIMEOUT_SEC = 120.0
# The viewer answers some requests with a temporary error while it loads an
# index or checks access ("the dataset index is loading, this can take a minute").
RETRY_DELAYS_SEC = (2, 5, 10, 20, 30)

_RESOLVED_NAMES: Dict[str, str] = {}
_DEFAULT_CONFIGS: Dict[str, str] = {}


class DatasetFetchError(RuntimeError):
    """The viewer or the Hub refused a request or answered something unusable."""


def _auth_headers() -> Dict[str, str]:
    token = huggingface_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _is_temporary(status: int) -> bool:
    # The viewer answers 500 for passing trouble too ("Unexpected error.", "the dataset
    # index is loading"); a request that keeps failing still fails after the retries.
    return status == 429 or status >= 500


def get_json(url: str) -> Any:
    """GET a JSON document, retrying the viewer's temporary errors."""
    attempts = len(RETRY_DELAYS_SEC) + 1
    for attempt in range(attempts):
        response = http_client.request(
            "GET", url, headers=_auth_headers(), timeout=REQUEST_TIMEOUT_SEC, use_env_proxy=True
        )
        if response.status < 400:
            try:
                return response.json()
            except ValueError as exc:
                raise DatasetFetchError(f"Not JSON from {url}: {exc}") from exc
        try:
            message = str(response.json().get("error") or response.text())
        except Exception:
            message = response.text()
        if attempt + 1 < attempts and _is_temporary(response.status):
            delay = RETRY_DELAYS_SEC[attempt]
            LOGGER.info("Temporary error %s from %s (%s); retrying in %ss", response.status, url, message, delay)
            time.sleep(delay)
            continue
        raise DatasetFetchError(f"HTTP {response.status} from {url}: {message}")
    raise DatasetFetchError(f"No answer from {url}")  # pragma: no cover


def resolve_dataset_name(name: str) -> str:
    """Return the dataset's current repository id; old short names were renamed on the Hub."""
    if name in _RESOLVED_NAMES:
        return _RESOLVED_NAMES[name]
    resolved = name
    if "/" not in name:
        # The Hub redirects a renamed dataset (gsm8k -> openai/gsm8k); urllib follows it.
        info = get_json(f"{HUB_URL}/api/datasets/{urllib.parse.quote(name, safe='')}")
        resolved = str(info.get("id") or name)
    _RESOLVED_NAMES[name] = resolved
    return resolved


def default_config(dataset: str) -> str:
    """The config ``load_dataset(name)`` picks when none is given: the only one, or 'default'."""
    if dataset in _DEFAULT_CONFIGS:
        return _DEFAULT_CONFIGS[dataset]
    payload = get_json(f"{VIEWER_URL}/splits?{urllib.parse.urlencode({'dataset': dataset})}")
    configs: List[str] = []
    for item in payload.get("splits", []):
        if item.get("config") not in configs:
            configs.append(item.get("config"))
    if len(configs) == 1:
        chosen = configs[0]
    elif "default" in configs:
        chosen = "default"
    else:
        raise DatasetFetchError(f"Dataset {dataset} has several configs ({', '.join(configs)}); name one.")
    _DEFAULT_CONFIGS[dataset] = chosen
    return chosen


def _rows_from_page(payload: Dict[str, Any], url: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in payload.get("rows", []):
        if item.get("truncated_cells"):
            raise DatasetFetchError(
                f"The viewer truncated cells {item['truncated_cells']} of row {item.get('row_idx')} at {url}; "
                "refusing to use a shortened row."
            )
        rows.append(item["row"])
    return rows


def _iter_pages(endpoint: str, params: Dict[str, str], page_size: int = PAGE_SIZE) -> Iterator[Dict[str, Any]]:
    offset = 0
    last_row_idx = -1
    while True:
        query = dict(params, offset=str(offset), length=str(max(1, min(PAGE_SIZE, page_size))))
        url = f"{VIEWER_URL}/{endpoint}?{urllib.parse.urlencode(query)}"
        payload = get_json(url)
        if payload.get("partial"):
            raise DatasetFetchError(
                f"The viewer only indexed part of {params.get('dataset')} for {url}; refusing a partial result."
            )
        items = payload.get("rows", [])
        for item in items:
            # Rows must come in dataset order, as load_dataset streamed them; /filter has no
            # documented default order, so a result out of order is refused, never reordered.
            row_idx = int(item.get("row_idx", last_row_idx + 1))
            if row_idx <= last_row_idx:
                raise DatasetFetchError(f"The viewer returned rows out of order at {url}; refusing them.")
            last_row_idx = row_idx
        rows = _rows_from_page(payload, url)
        yield from rows
        offset += len(rows)
        total = int(payload.get("num_rows_total") or 0)
        if not rows or offset >= total:
            return


def iter_rows(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    page_size: int = PAGE_SIZE,
) -> Iterator[Dict[str, Any]]:
    """Yield a split's rows in order, a page (at most 100) at a time, only as they are consumed."""
    dataset = resolve_dataset_name(dataset_name)
    config = config_name or default_config(dataset)
    yield from _iter_pages("rows", {"dataset": dataset, "config": config, "split": split}, page_size)


def iter_filtered_rows(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    where: str,
    page_size: int = PAGE_SIZE,
) -> Iterator[Dict[str, Any]]:
    """Yield the rows matching a viewer ``where`` clause (SQL on the split's columns), in row order."""
    dataset = resolve_dataset_name(dataset_name)
    config = config_name or default_config(dataset)
    yield from _iter_pages("filter", {"dataset": dataset, "config": config, "split": split, "where": where}, page_size)


def download_bytes(url: str) -> bytes:
    """Download one file over HTTPS (with the token when the URL is on the Hub)."""
    headers = _auth_headers() if url.startswith(HUB_URL) else {}
    response = http_client.request("GET", url, headers=headers, timeout=REQUEST_TIMEOUT_SEC, use_env_proxy=True)
    if response.status >= 400:
        raise DatasetFetchError(f"HTTP {response.status} from {url}: {response.text()[:300]}")
    return response.body
