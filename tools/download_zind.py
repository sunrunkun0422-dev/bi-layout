#!/usr/bin/env python3
"""Download ZInD into the layout expected by this project.

This mirrors the official Zillow download flow, but avoids optional progress
dependencies so it can run in lean Python environments.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from getpass import getpass
from pathlib import Path

import requests


BRIDGE_API_URL = "https://api.bridgedataoutput.com/api/v2/OData/zgindoor/Indoor/replication"
RAW_BASE_URL = "https://raw.githubusercontent.com/zillow/zind/main"
METADATA_FILES = ("zind_partition.json", "room_shape_simplicity_labels.json")
MAX_RETRIES = 3
JSON_TIMEOUT = 300
IMAGE_TIMEOUT = 60


LOGGER = logging.getLogger("download_zind")
LINK_NEXT_RE = re.compile(r"<([^>]+)>;\s*rel=\"next\"")


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("w") as fh:
        json.dump(value, fh)


def read_json(path: Path):
    with path.open("r") as fh:
        return json.load(fh)


def load_json_if_valid(path: Path):
    try:
        return read_json(path)
    except Exception as exc:
        LOGGER.info("Cached JSON is incomplete or invalid: %s", exc)
        return None


def get_next_link(response, payload):
    link_header = response.headers.get("link", "")
    match = LINK_NEXT_RE.search(link_header)
    if match:
        return match.group(1)
    return payload.get("@odata.nextLink") or payload.get("odata.nextLink")


def request_bridge_page(url: str, headers: dict, dest_path: Path, timeout: int):
    request_headers = dict(headers)
    request_headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "curl/8.0.0",
        }
    )
    response = requests.get(
        url,
        headers=request_headers,
        stream=True,
        timeout=(30, timeout),
    )
    if response.status_code in (401, 403):
        raise RuntimeError("Bridge API rejected the token. Please check approval and Server Token.")
    response.raise_for_status()

    bytes_read = 0
    content_length = int(response.headers.get("content-length", 0))
    next_log_at = 50 * 1024 * 1024
    with dest_path.open("wb") as fh:
        for chunk in response.iter_content(1024 * 1024):
            if not chunk:
                continue
            fh.write(chunk)
            bytes_read += len(chunk)
            if bytes_read >= next_log_at:
                total = f"/{content_length / 1024 / 1024:.1f} MB" if content_length else ""
                LOGGER.info("Downloaded %s: %.1f MB%s", dest_path.name, bytes_read / 1024 / 1024, total)
                next_log_at += 50 * 1024 * 1024

    payload = read_json(dest_path)
    return payload, get_next_link(response, payload)


def download_metadata_file(output_dir: Path, filename: str) -> None:
    dest = output_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        LOGGER.info("Metadata exists: %s", dest)
        return

    url = f"{RAW_BASE_URL}/{filename}"
    LOGGER.info("Downloading metadata: %s", filename)
    response = requests.get(url, timeout=JSON_TIMEOUT)
    response.raise_for_status()
    dest.write_bytes(response.content)


def download_metadata(output_dir: Path) -> None:
    for filename in METADATA_FILES:
        download_metadata_file(output_dir, filename)


def get_zind_houses(output_dir: Path, server_token: str):
    headers = {"Authorization": f"Bearer {server_token}"}
    pages_dir = output_dir / "bridge_pages"
    mkdir(pages_dir)

    houses = []
    page_index = 0
    next_url = BRIDGE_API_URL
    while next_url:
        page_path = pages_dir / f"zind_response_page_{page_index:04d}.json"
        cached = load_json_if_valid(page_path) if page_path.exists() else None
        if cached and "value" in cached:
            payload = cached
            next_url = payload.get("_cached_next_url")
            LOGGER.info("Loaded cached Bridge page %d with %d homes", page_index, len(payload["value"]))
        else:
            payload = None
            for attempt in range(1, MAX_RETRIES + 1):
                LOGGER.info(
                    "Requesting ZInD index page %d from Bridge API (attempt %d/%d)",
                    page_index,
                    attempt,
                    MAX_RETRIES,
                )
                try:
                    payload, next_url = request_bridge_page(
                        next_url,
                        headers,
                        page_path,
                        timeout=JSON_TIMEOUT * attempt,
                    )
                    if "value" not in payload:
                        raise RuntimeError("Bridge response did not contain a 'value' field.")
                    payload["_cached_next_url"] = next_url
                    write_json(page_path, payload)
                    break
                except Exception as exc:
                    if attempt == MAX_RETRIES:
                        raise
                    LOGGER.warning("Bridge page request failed: %s; retrying", exc)
                    time.sleep(15 * attempt)

        houses.extend(payload["value"])
        page_index += 1

    response_path = output_dir / "zind_response.json"
    write_json(response_path, {"value": houses})
    LOGGER.info("Loaded Bridge index: %d homes across %d page(s)", len(houses), page_index)
    return houses


def keep_required_keys(details: dict) -> dict:
    keys = {
        "merger",
        "redraw",
        "scale_meters_per_coordinate",
        "floorplan_to_redraw_transformation",
    }
    return {key: value for key, value in details.items() if key in keys}


def process_house(details: dict, output_dir: Path):
    house_id = details["home_id"]
    house_dir = output_dir / house_id
    panos_dir = house_dir / "panos"
    floor_plans_dir = house_dir / "floor_plans"
    mkdir(panos_dir)
    mkdir(floor_plans_dir)

    local_details = dict(details)
    downloads = []

    for floor_name, floor_details in details["merger"].items():
        for complete_room_name, complete_room_details in floor_details.items():
            for partial_room_name, partial_room_details in complete_room_details.items():
                for pano_name, pano_details in partial_room_details.items():
                    pano_filename = f"{floor_name}_{partial_room_name}_{pano_name}.jpg"
                    pano_dest = panos_dir / pano_filename
                    downloads.append((pano_details["image_path"], str(pano_dest), pano_details["checksum"]))
                    local_details["merger"][floor_name][complete_room_name][partial_room_name][pano_name][
                        "image_path"
                    ] = f"panos/{pano_filename}"

    for floor_name, floor_details in details["floorplan_to_redraw_transformation"].items():
        floor_plan_rel = Path("floor_plans") / f"{floor_name}.png"
        floor_plan_dest = house_dir / floor_plan_rel
        downloads.append((floor_details["image_path"], str(floor_plan_dest), floor_details["checksum"]))
        local_details["floorplan_to_redraw_transformation"][floor_name]["image_path"] = str(floor_plan_rel)

    write_json(house_dir / "zind_data.json", keep_required_keys(local_details))
    return downloads


def create_download_list(output_dir: Path, houses, partial_percentage: float):
    status_path = output_dir / "download_status.json"
    if status_path.exists():
        status = read_json(status_path)
        if status.get("partial_download_percentage") == partial_percentage and "files_list" in status:
            LOGGER.info("Using existing download list from %s", status_path)
            return status["files_list"]

    selected_count = max(1, int(len(houses) * partial_percentage / 100.0))
    selected_houses = houses[:selected_count]
    LOGGER.info("Preparing folder structure for %d/%d homes", selected_count, len(houses))

    downloads = []
    for index, house in enumerate(selected_houses, start=1):
        downloads.extend(process_house(house, output_dir))
        if index % 50 == 0 or index == selected_count:
            LOGGER.info("Prepared %d/%d homes", index, selected_count)

    write_json(
        status_path,
        {"partial_download_percentage": partial_percentage, "files_list": downloads},
    )
    return downloads


def files_to_download(downloads):
    pending = []
    for url, dest, checksum in downloads:
        dest_path = Path(dest)
        if not dest_path.exists() or md5sum(dest_path) != checksum:
            pending.append((url, dest, checksum))
    return pending


def download_one(item):
    url, dest, checksum = item
    dest_path = Path(dest)
    mkdir(dest_path.parent)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, stream=True, timeout=IMAGE_TIMEOUT * attempt)
            response.raise_for_status()
            tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
            with tmp_path.open("wb") as fh:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            if md5sum(tmp_path) != checksum:
                raise RuntimeError("checksum mismatch")
            tmp_path.replace(dest_path)
            return True, dest
        except Exception as exc:
            if attempt == MAX_RETRIES:
                return False, f"{dest}: {exc}"
            time.sleep(10 * attempt)

    return False, dest


def download_all(output_dir: Path, server_token: str, workers: int, partial_percentage: float) -> int:
    mkdir(output_dir)
    download_metadata(output_dir)
    houses = get_zind_houses(output_dir, server_token)
    downloads = create_download_list(output_dir, houses, partial_percentage)
    pending = files_to_download(downloads)

    LOGGER.info("Images/floor plans pending: %d/%d", len(pending), len(downloads))
    if not pending:
        LOGGER.info("All files already downloaded and verified")
        return 0

    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, item) for item in pending]
        for future in as_completed(futures):
            ok, message = future.result()
            completed += 1
            if not ok:
                failures.append(message)
            if completed % 100 == 0 or completed == len(pending):
                LOGGER.info("Downloaded/checked %d/%d pending files", completed, len(pending))

    if failures:
        LOGGER.error("Failed files: %d. Re-run the same command to resume.", len(failures))
        for failure in failures[:20]:
            LOGGER.error("  %s", failure)
        return 1

    LOGGER.info("ZInD download complete and verified: %s", output_dir)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Download ZInD data for Bi_Layout")
    parser.add_argument("-o", "--output-dir", default="src/dataset/ZInD")
    parser.add_argument("-s", "--server-token", default=os.environ.get("ZIND_SERVER_TOKEN"))
    parser.add_argument("-n", "--workers", type=int, default=min(16, (os.cpu_count() or 4)))
    parser.add_argument("-p", "--partial-download-percentage", type=float, default=100.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    token = args.server_token or getpass("Bridge Server Token: ")
    if not token:
        LOGGER.error("Missing Bridge Server Token.")
        return 2

    try:
        return download_all(
            output_dir=Path(args.output_dir),
            server_token=token,
            workers=args.workers,
            partial_percentage=args.partial_download_percentage,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
