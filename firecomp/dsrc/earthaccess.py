"""Earthaccess helper utilities for safe concurrent downloads.

This module provides utilities for downloading files from NASA Earthdata
with support for:
- Concurrent job safety (using .pending files to avoid race conditions)
- File integrity verification (checking size against expected)
- Partial download detection (for cancelled jobs)
"""

import os
import time
from typing import Optional


def get_expected_size(result) -> Optional[int]:
    """Get expected file size in bytes from earthaccess result.

    Args:
        result: Earthaccess search result object

    Returns:
        Expected file size in bytes, or None if not available.
    """
    try:
        umm = result["umm"]
        archive_info = umm["DataGranule"]["ArchiveAndDistributionInformation"]
        return archive_info[0]["SizeInBytes"]
    except (KeyError, IndexError, TypeError):
        return None


def verify_file_size(local_path: str, result) -> bool:
    """Check if local file size matches expected size from earthaccess result.

    Args:
        local_path: Path to local file
        result: Earthaccess search result object

    Returns:
        True if sizes match, False otherwise.
        Returns True if expected size is not available (can't verify).
    """
    if not os.path.exists(local_path):
        return False

    expected_size = get_expected_size(result)
    if expected_size is None:
        # Can't verify, assume OK
        return True

    actual_size = os.path.getsize(local_path)
    return actual_size == expected_size


def is_download_pending(local_path: str) -> bool:
    """Check if a .pending file exists (another job is downloading).

    Args:
        local_path: Path to the target file

    Returns:
        True if .pending file exists.
    """
    pending_path = local_path + ".pending"
    return os.path.exists(pending_path)


def create_pending_marker(local_path: str) -> str:
    """Create a .pending marker file.

    Args:
        local_path: Path to the target file

    Returns:
        Path to the created .pending file.
    """
    pending_path = local_path + ".pending"
    with open(pending_path, "w") as f:
        f.write(f"pid={os.getpid()}\ntime={time.time()}\n")
    return pending_path


def remove_pending_marker(local_path: str) -> None:
    """Remove the .pending marker file.

    Args:
        local_path: Path to the target file
    """
    pending_path = local_path + ".pending"
    if os.path.exists(pending_path):
        os.remove(pending_path)


def file_status(local_path: str, result) -> str:
    """Check status of a file for download.

    Args:
        local_path: Path to the target file
        result: Earthaccess search result object

    Returns:
        One of:
        - "valid": File exists and size matches
        - "invalid": File exists but size doesn't match (corrupted/partial)
        - "pending": Another job is downloading
        - "missing": File doesn't exist
    """
    # Check pending FIRST - if download is in progress, file may exist but be incomplete
    if is_download_pending(local_path):
        return "pending"
    elif os.path.exists(local_path):
        if verify_file_size(local_path, result):
            return "valid"
        else:
            return "invalid"
    else:
        return "missing"


def safe_download(
    results: list,
    output_dir: str,
    delete_invalid: bool = True,
    max_pending_wait: float = 10 * 60,  # 10 minutes
) -> dict[str, str]:
    """Download files with .pending file locking for concurrent job safety.

    Args:
        results: List of earthaccess search results to download
        output_dir: Directory to download files to
        delete_invalid: If True, delete files with incorrect size before re-downloading
        max_pending_wait: Max seconds to wait for pending downloads at end (default 5min) or <= 0 to not wait

    Returns:
        Dict mapping granule prefix to local file path for successfully downloaded
        or already-existing valid files.
    """
    import earthaccess
    from firecomp.dsrc.viirs import result_filename, viirs_granule_prefix

    os.makedirs(output_dir, exist_ok=True)

    prefix_to_path: dict[str, str] = {}
    to_download: list = []  # (r, prefix, local_path, pending_path)
    pending_from_others: list = (
        []
    )  # (r, prefix, local_path) - files other jobs are downloading

    # Phase 1: Check status, create pending markers immediately for files we'll download
    for r in results:
        fname = result_filename(r)
        prefix = viirs_granule_prefix(fname)
        local_path = os.path.join(output_dir, fname)

        status = file_status(local_path, r)

        if status == "valid":
            if prefix:
                prefix_to_path[prefix] = local_path
        elif status == "invalid":
            if delete_invalid:
                print(f"  Deleting invalid/partial file: {fname}")
                os.remove(local_path)
                # Create pending marker immediately
                pending_path = create_pending_marker(local_path)
                to_download.append((r, prefix, local_path, pending_path))
            else:
                print(f"  Skipping invalid file (delete_invalid=False): {fname}")
        elif status == "pending":
            # Track for waiting at the end
            pending_from_others.append((r, prefix, local_path))
            print(f"  Pending (another job downloading): {fname}")
        else:  # missing
            # Create pending marker immediately
            pending_path = create_pending_marker(local_path)
            to_download.append((r, prefix, local_path, pending_path))

    # Phase 2: Download our files
    if to_download:
        try:
            download_list = [r for r, _, _, _ in to_download]
            print(f"  Downloading {len(download_list)} files...")
            earthaccess.download(download_list, output_dir)
        finally:
            # Remove our pending markers and verify
            for r, prefix, local_path, pending_path in to_download:
                remove_pending_marker(local_path)
                if os.path.exists(local_path) and verify_file_size(local_path, r):
                    if prefix:
                        prefix_to_path[prefix] = local_path
                else:
                    print(
                        f"  Warning: Downloaded file missing or wrong size: {local_path}"
                    )

    # Phase 3: Wait for pending files from other jobs (at the END)
    if pending_from_others and max_pending_wait > 0:
        print(
            f"  Waiting for {len(pending_from_others)} pending downloads from other jobs..."
        )
        for r, prefix, local_path in pending_from_others:
            waited = 0
            while is_download_pending(local_path) and waited < max_pending_wait:
                time.sleep(1)
                waited += 1
            # Check if file is now valid
            if os.path.exists(local_path) and verify_file_size(local_path, r):
                if prefix:
                    prefix_to_path[prefix] = local_path
            else:
                fname = result_filename(r)
                print(f"  Warning: Pending file not ready after wait: {fname}")

    return prefix_to_path
