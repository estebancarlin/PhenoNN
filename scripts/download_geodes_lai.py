#!/usr/bin/env python3
"""Download regular (non-GCM) GEOV2 AVHRR LAI archives from GEODES."""

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests


SEARCH_URL = "https://geodes-portal.cnes.fr/api/stac/items"
DATASET = "THEIA_POSTEL_VEGETATION_LAI"
SERIES = {
    "R01": (dt.date(1982, 1, 5), dt.date(2018, 12, 25)),
    "R02": (dt.date(2017, 1, 5), dt.date(2022, 12, 25)),
}
SIZE_RE = re.compile(r"File size:\s*(\d+)")
MD5_RE = re.compile(r"Checksum MD5:\s*([0-9a-fA-F]{32})")


def request_with_retry(session, method, url, **kwargs):
    attempt = 0
    while attempt < 10:
        try:
            response = session.request(method, url, timeout=(30, 180), **kwargs)
            if response.status_code == 403 and "exceeds quota limit" in response.text:
                print(
                    f"{dt.datetime.now().isoformat(timespec='seconds')} "
                    "GEODES hourly download quota reached; waiting 61 minutes",
                    flush=True,
                )
                response.close()
                time.sleep(61 * 60)
                attempt = 0
                continue
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(300, 2**attempt)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 9:
                raise
            delay = min(300, 2**attempt)
        time.sleep(delay + random.random())
        attempt += 1
    raise RuntimeError(f"Request failed after retries: {url}")


def expected_dates(start, end):
    dates = []
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            for day in (5, 15, 25):
                value = dt.date(year, month, day)
                if start <= value <= end:
                    dates.append(value)
    return dates


def inventory_series(session, release):
    start, end = SERIES[release]
    prefix = f"THEIA_GEOV2_{release}_AVHRR_LAI_"
    identifier_re = re.compile(rf"^{re.escape(prefix)}(\d{{8}})$")
    query = {
        "dataset": {"in": [DATASET]},
        "identifier": {"contains": prefix},
        "start_datetime": {
            "gte": f"{start.isoformat()}T00:00:00Z",
            "lte": f"{end.isoformat()}T23:59:59Z",
        },
    }
    items = []
    page = 1
    matched = None
    while matched is None or len(items) < matched:
        body = {
            "page": page,
            "limit": 80,
            "sortBy": [{"direction": "asc", "field": "start_datetime"}],
            "query": query,
        }
        response = request_with_retry(session, "POST", SEARCH_URL, json=body)
        payload = response.json()
        matched = int(payload["context"]["matched"])
        features = payload.get("features", [])
        if not features:
            break
        items.extend(features)
        page += 1
        time.sleep(0.1)

    records = []
    for item in items:
        identifier = item.get("properties", {}).get("identifier", "")
        match = identifier_re.fullmatch(identifier)
        if not match:
            raise RuntimeError(f"Unexpected identifier in result: {identifier!r}")
        assets = item.get("assets", {})
        if len(assets) != 1:
            raise RuntimeError(
                f"Expected one asset for {identifier}, got {len(assets)}"
            )
        filename, asset = next(iter(assets.items()))
        description = asset.get("description", "")
        size_match = SIZE_RE.search(description)
        md5_match = MD5_RE.search(description)
        if not size_match or not md5_match:
            raise RuntimeError(f"Missing size or MD5 metadata for {identifier}")
        if "Is online: true" not in description:
            raise RuntimeError(f"Asset is not online: {identifier}")
        records.append(
            {
                "release": release,
                "date": dt.datetime.strptime(match.group(1), "%Y%m%d")
                .date()
                .isoformat(),
                "identifier": identifier,
                "item_id": item["id"],
                "filename": filename,
                "url": asset["href"],
                "size": int(size_match.group(1)),
                "md5": md5_match.group(1).lower(),
            }
        )

    actual = [dt.date.fromisoformat(record["date"]) for record in records]
    expected = expected_dates(start, end)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"Incomplete {release} inventory: missing={missing[:10]}, extra={extra[:10]}"
        )
    return records


def file_md5(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def encoded_download_url(url):
    prefix, separator, suffix = url.partition("/api/download/")
    if not separator:
        return url
    item_id, files_separator, file_id = suffix.partition("/files/")
    if not files_separator:
        return url
    return f"{prefix}{separator}{quote(item_id, safe='')}/files/{file_id}"


def download_record(session, record, output_dir, api_key):
    destination = output_dir / record["release"] / record["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if destination.exists():
        if (
            destination.stat().st_size == record["size"]
            and file_md5(destination) == record["md5"]
        ):
            return "skipped"
        destination.replace(destination.with_name(destination.name + ".invalid"))

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > record["size"]:
        partial.unlink()
        offset = 0

    headers = {"X-API-Key": api_key}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    response = request_with_retry(
        session,
        "GET",
        encoded_download_url(record["url"]),
        headers=headers,
        stream=True,
    )
    if offset and response.status_code != 206:
        partial.unlink(missing_ok=True)
        offset = 0
    mode = "ab" if offset and response.status_code == 206 else "wb"
    with partial.open(mode) as stream:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                stream.write(chunk)

    if partial.stat().st_size != record["size"]:
        raise RuntimeError(
            f"Size mismatch for {record['filename']}: "
            f"{partial.stat().st_size} != {record['size']}"
        )
    checksum = file_md5(partial)
    if checksum != record["md5"]:
        partial.unlink()
        raise RuntimeError(
            f"MD5 mismatch for {record['filename']}: {checksum} != {record['md5']}"
        )
    partial.replace(destination)
    return "downloaded"


def save_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("GEODES_API_KEY", "")
    if not args.inventory_only and not api_key:
        parser.error("GEODES_API_KEY must be set for downloads")

    session = requests.Session()
    all_records = []
    for release in SERIES:
        records = inventory_series(session, release)
        all_records.extend(records)
        total = sum(record["size"] for record in records)
        print(f"{release}: {len(records)} files, {total / 1024**3:.3f} GiB", flush=True)
    save_manifest(args.output_dir / "manifest.json", all_records)
    if args.inventory_only:
        return

    downloaded = skipped = 0
    total_records = len(all_records)
    for index, record in enumerate(all_records, start=1):
        while True:
            try:
                status = download_record(session, record, args.output_dir, api_key)
                break
            except (requests.RequestException, RuntimeError, OSError) as error:
                print(
                    f"{dt.datetime.now().isoformat(timespec='seconds')} "
                    f"retrying {record['filename']} in 60 seconds: {error}",
                    flush=True,
                )
                time.sleep(60)
        downloaded += status == "downloaded"
        skipped += status == "skipped"
        print(
            f"[{index}/{total_records}] {status}: {record['filename']} "
            f"(downloaded={downloaded}, skipped={skipped})",
            flush=True,
        )
        time.sleep(0.25)
    print(f"Complete: downloaded={downloaded}, skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
