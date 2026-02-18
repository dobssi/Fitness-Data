"""
ci/dropbox_sync.py — Upload/download pipeline files to/from Dropbox
===================================================================
Used by GitHub Actions to persist pipeline state between runs.

Usage:
  python ci/dropbox_sync.py download --items file1.xlsx file2.csv
  python ci/dropbox_sync.py upload --items file1.xlsx file2.csv
  python ci/dropbox_sync.py download --cache-dir persec_cache_FULL
  python ci/dropbox_sync.py upload --cache-dir persec_cache_FULL
  python ci/dropbox_sync.py download --cache-dir persec_cache_FULL --cache-full  # tar.gz mode

Cache sync modes:
  --cache-dir DIR             Incremental: upload/download only new .npz files (default)
  --cache-dir DIR --cache-full  Full tar.gz pack/unpack (for rebuilds or first-time setup)

Authentication (refresh token — recommended, never expires):
  DROPBOX_REFRESH_TOKEN — long-lived refresh token
  DROPBOX_APP_KEY       — app key from Dropbox App Console
  DROPBOX_APP_SECRET    — app secret from Dropbox App Console

Authentication (legacy — expires after 4 hours):
  DROPBOX_TOKEN — short-lived OAuth2 access token
  
Dropbox folder layout:
  /Running and Cycling/DataPipeline/
    ├── Master_FULL_GPSQ_ID.xlsx
    ├── Master_FULL_GPSQ_ID_post.xlsx
    ├── athlete_data.csv
    ├── re_model_s4_FULL.json
    ├── sync_state.json
    ├── TotalHistory.zip
    └── cache/
        ├── persec_cache_FULL.tar.gz          (full archive, for --cache-full)
        └── persec_cache_FULL/                (individual .npz files, for incremental)
            ├── 524737937055_token_xxx.npz
            ├── 524737937055_token_xxx_summary.json
            └── ...
"""

import os
import sys
import tarfile
import argparse
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests'")


DROPBOX_BASE = "/Running and Cycling/DataPipeline"
UPLOAD_URL = "https://content.dropboxapi.com/2/files/upload"
DOWNLOAD_URL = "https://content.dropboxapi.com/2/files/download"
TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"

# Dropbox single-request upload limit: 150 MB
# For larger files, use upload_session (chunked) — supports up to 350 GB
MAX_SINGLE_UPLOAD = 150 * 1024 * 1024
CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB chunks for session uploads


def get_token() -> str:
    """Get a valid Dropbox access token.
    
    Prefers refresh token flow (never expires) over bare access token.
    """
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN", "")
    app_key = os.environ.get("DROPBOX_APP_KEY", "")
    app_secret = os.environ.get("DROPBOX_APP_SECRET", "")

    if refresh_token and app_key and app_secret:
        resp = requests.post(TOKEN_URL, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": app_key,
            "client_secret": app_secret,
        }, timeout=30)
        if resp.status_code == 200:
            token = resp.json().get("access_token", "")
            if token:
                print("  ✓ Obtained fresh access token via refresh token")
                return token
        print(f"  ✗ Refresh token exchange failed ({resp.status_code}): {resp.text[:200]}")
        sys.exit("ERROR: Could not obtain access token from refresh token")

    # Fallback: bare access token (expires after 4 hours)
    token = os.environ.get("DROPBOX_TOKEN", "")
    if token:
        print("  ⚠ Using bare access token (may expire). Set DROPBOX_REFRESH_TOKEN for reliability.")
        return token

    sys.exit(
        "ERROR: No Dropbox credentials found. Set either:\n"
        "  - DROPBOX_REFRESH_TOKEN + DROPBOX_APP_KEY + DROPBOX_APP_SECRET (recommended)\n"
        "  - DROPBOX_TOKEN (legacy, expires after 4 hours)"
    )


def dropbox_upload(local_path: str, remote_path: str, token: str):
    """Upload a single file to Dropbox. Uses chunked upload for files > 150 MB."""
    if not os.path.exists(local_path):
        print(f"  ⚠ Skip (not found): {local_path}")
        return False
    
    size = os.path.getsize(local_path)
    
    if size > MAX_SINGLE_UPLOAD:
        return _dropbox_upload_chunked(local_path, remote_path, token, size)
    
    import json
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
        "Dropbox-API-Arg": json.dumps({
            "path": remote_path,
            "mode": "overwrite",
            "autorename": False,
            "mute": True,
        }),
    }
    
    with open(local_path, "rb") as f:
        resp = requests.post(UPLOAD_URL, headers=headers, data=f, timeout=120)
    
    if resp.status_code == 200:
        print(f"  ✓ Uploaded: {local_path} → {remote_path} ({size / 1024:.0f} KB)")
        return True
    else:
        print(f"  ✗ Upload failed ({resp.status_code}): {local_path}")
        print(f"    {resp.text[:200]}")
        return False


def _dropbox_upload_chunked(local_path: str, remote_path: str, token: str, size: int):
    """Upload a large file using Dropbox upload sessions (chunked)."""
    import json
    
    SESSION_START = "https://content.dropboxapi.com/2/files/upload_session/start"
    SESSION_APPEND = "https://content.dropboxapi.com/2/files/upload_session/append_v2"
    SESSION_FINISH = "https://content.dropboxapi.com/2/files/upload_session/finish"
    
    chunks = (size + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  ↑ Chunked upload ({size / 1024 / 1024:.0f} MB, {chunks} chunks): {local_path}")
    
    try:
        with open(local_path, "rb") as f:
            # Start session
            chunk = f.read(CHUNK_SIZE)
            resp = requests.post(
                SESSION_START,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/octet-stream",
                    "Dropbox-API-Arg": json.dumps({"close": False}),
                },
                data=chunk,
                timeout=120,
            )
            if resp.status_code != 200:
                print(f"  ✗ Session start failed ({resp.status_code}): {resp.text[:200]}")
                return False
            
            session_id = resp.json()["session_id"]
            offset = len(chunk)
            chunk_num = 1
            
            # Append remaining chunks
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                chunk_num += 1
                print(f"    chunk {chunk_num}/{chunks}...")
                
                resp = requests.post(
                    SESSION_APPEND,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/octet-stream",
                        "Dropbox-API-Arg": json.dumps({
                            "cursor": {"session_id": session_id, "offset": offset},
                            "close": False,
                        }),
                    },
                    data=chunk,
                    timeout=120,
                )
                if resp.status_code != 200:
                    print(f"  ✗ Append failed at chunk {chunk_num} ({resp.status_code}): {resp.text[:200]}")
                    return False
                offset += len(chunk)
            
            # Finish session
            resp = requests.post(
                SESSION_FINISH,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/octet-stream",
                    "Dropbox-API-Arg": json.dumps({
                        "cursor": {"session_id": session_id, "offset": offset},
                        "commit": {
                            "path": remote_path,
                            "mode": "overwrite",
                            "autorename": False,
                            "mute": True,
                        },
                    }),
                },
                data=b"",
                timeout=120,
            )
            if resp.status_code == 200:
                print(f"  ✓ Uploaded (chunked): {local_path} → {remote_path} ({size / 1024 / 1024:.0f} MB)")
                return True
            else:
                print(f"  ✗ Session finish failed ({resp.status_code}): {resp.text[:200]}")
                return False
    except Exception as e:
        print(f"  ✗ Chunked upload error: {repr(e)}")
        return False


def dropbox_download(remote_path: str, local_path: str, token: str):
    """Download a single file from Dropbox."""
    import json
    headers = {
        "Authorization": f"Bearer {token}",
        "Dropbox-API-Arg": json.dumps({"path": remote_path}),
    }
    
    resp = requests.post(DOWNLOAD_URL, headers=headers, timeout=120, stream=True)
    
    if resp.status_code == 200:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size = os.path.getsize(local_path)
        print(f"  ✓ Downloaded: {remote_path} → {local_path} ({size / 1024:.0f} KB)")
        return True
    elif resp.status_code == 409:
        # File not found — this is OK for first run
        print(f"  ⚠ Not found on Dropbox: {remote_path}")
        return False
    else:
        print(f"  ✗ Download failed ({resp.status_code}): {remote_path}")
        print(f"    {resp.text[:200]}")
        return False


def pack_cache(cache_dir: str, archive_path: str):
    """Pack persec cache directory into tar.gz for efficient transfer."""
    if not os.path.isdir(cache_dir):
        print(f"  ⚠ Cache directory not found: {cache_dir}")
        return False
    
    npz_files = list(Path(cache_dir).glob("*.npz"))
    print(f"  Packing {len(npz_files)} .npz files from {cache_dir}...")
    
    with tarfile.open(archive_path, "w:gz") as tar:
        for f in npz_files:
            tar.add(f, arcname=f.name)
    
    size = os.path.getsize(archive_path)
    print(f"  ✓ Packed: {archive_path} ({size / 1024 / 1024:.1f} MB)")
    return True


def unpack_cache(archive_path: str, cache_dir: str):
    """Unpack tar.gz into persec cache directory."""
    if not os.path.exists(archive_path):
        print(f"  ⚠ Cache archive not found: {archive_path}")
        return False
    
    os.makedirs(cache_dir, exist_ok=True)
    
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(cache_dir)
    
    npz_count = len(list(Path(cache_dir).glob("*.npz")))
    print(f"  ✓ Unpacked {npz_count} .npz files to {cache_dir}")
    return True


# --- Incremental cache sync ---

LIST_FOLDER_URL = "https://api.dropboxapi.com/2/files/list_folder"
LIST_FOLDER_CONTINUE_URL = "https://api.dropboxapi.com/2/files/list_folder/continue"


def dropbox_list_folder(remote_dir: str, token: str) -> dict:
    """List files in a Dropbox folder. Returns {filename: size} dict.
    
    Returns empty dict if folder doesn't exist.
    """
    import json
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    files = {}
    body = {"path": remote_dir, "recursive": False, "limit": 2000}
    
    resp = requests.post(LIST_FOLDER_URL, headers=headers, json=body, timeout=60)
    
    if resp.status_code == 409:
        # Folder doesn't exist yet — that's OK
        detail = resp.text
        if "not_found" in detail:
            print(f"  ⚠ Remote folder not found: {remote_dir} (will be created on first upload)")
            return {}
        print(f"  ✗ List folder error ({resp.status_code}): {detail[:200]}")
        return {}
    elif resp.status_code != 200:
        print(f"  ✗ List folder failed ({resp.status_code}): {resp.text[:200]}")
        return {}
    
    data = resp.json()
    for entry in data.get("entries", []):
        if entry[".tag"] == "file":
            files[entry["name"]] = entry["size"]
    
    # Handle pagination
    while data.get("has_more", False):
        resp = requests.post(
            LIST_FOLDER_CONTINUE_URL,
            headers=headers,
            json={"cursor": data["cursor"]},
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"  ✗ List folder continue failed ({resp.status_code})")
            break
        data = resp.json()
        for entry in data.get("entries", []):
            if entry[".tag"] == "file":
                files[entry["name"]] = entry["size"]
    
    return files


def sync_cache_incremental_upload(cache_dir: str, base: str, token: str) -> tuple:
    """Upload only new/changed cache files (NPZ + summary JSON) to Dropbox.
    
    Compares local files against remote listing. Uploads files that are
    missing remotely or differ in size.
    
    Returns (uploaded_count, skipped_count).
    """
    if not os.path.isdir(cache_dir):
        print(f"  ⚠ Cache directory not found: {cache_dir}")
        return (0, 0)
    
    cache_name = os.path.basename(cache_dir)
    remote_dir = f"{base}/{cache_name}"
    
    # List local cache files (.npz + _summary.json + _cache_index.json)
    local_files = {}
    for f in Path(cache_dir).iterdir():
        if f.suffix in (".npz", ".json"):
            local_files[f.name] = f.stat().st_size
    
    if not local_files:
        print(f"  ⚠ No cache files found in {cache_dir}")
        return (0, 0)
    
    print(f"  Local cache: {len(local_files)} files")
    
    # List remote cache files
    remote_files = dropbox_list_folder(remote_dir, token)
    print(f"  Remote cache: {len(remote_files)} files")
    
    # Find files to upload (missing remotely or different size)
    to_upload = []
    for name, local_size in local_files.items():
        remote_size = remote_files.get(name)
        if remote_size is None or remote_size != local_size:
            to_upload.append(name)
    
    if not to_upload:
        print(f"  ✓ Cache in sync — nothing to upload")
        return (0, len(local_files))
    
    print(f"  Uploading {len(to_upload)} new/changed cache files...")
    uploaded = 0
    for name in to_upload:
        local_path = os.path.join(cache_dir, name)
        remote_path = f"{remote_dir}/{name}"
        if dropbox_upload(local_path, remote_path, token):
            uploaded += 1
    
    skipped = len(local_files) - len(to_upload)
    print(f"  ✓ Cache upload: {uploaded} uploaded, {skipped} already in sync")
    return (uploaded, skipped)


def sync_cache_incremental_download(cache_dir: str, base: str, token: str) -> tuple:
    """Download only missing cache files (NPZ + summary JSON) from Dropbox.
    
    Compares remote files against local listing. Downloads files that are
    missing locally or differ in size.
    
    Falls back to tar.gz download if no incremental files exist on Dropbox.
    
    Returns (downloaded_count, skipped_count).
    """
    cache_name = os.path.basename(cache_dir)
    remote_dir = f"{base}/{cache_name}"
    
    # List remote cache files
    remote_files = dropbox_list_folder(remote_dir, token)
    if not remote_files:
        print(f"  ⚠ No remote cache files found — falling back to tar.gz if available")
        # Try tar.gz fallback
        archive = f"{cache_dir}.tar.gz"
        remote_archive = f"{base}/cache/{os.path.basename(archive)}"
        if dropbox_download(remote_archive, archive, token):
            unpack_cache(archive, cache_dir)
            os.remove(archive)
            return (1, 0)
        return (0, 0)
    
    print(f"  Remote cache: {len(remote_files)} files")
    
    # List local cache files
    os.makedirs(cache_dir, exist_ok=True)
    local_files = {}
    for f in Path(cache_dir).iterdir():
        if f.suffix in (".npz", ".json"):
            local_files[f.name] = f.stat().st_size
    
    print(f"  Local cache: {len(local_files)} files")
    
    # Find files to download (missing locally or different size)
    to_download = []
    for name, remote_size in remote_files.items():
        local_size = local_files.get(name)
        if local_size is None or local_size != remote_size:
            to_download.append(name)
    
    if not to_download:
        print(f"  ✓ Cache in sync — nothing to download")
        return (0, len(remote_files))
    
    print(f"  Downloading {len(to_download)} missing/changed cache files...")
    downloaded = 0
    for name in to_download:
        remote_path = f"{remote_dir}/{name}"
        local_path = os.path.join(cache_dir, name)
        if dropbox_download(remote_path, local_path, token):
            downloaded += 1
    
    skipped = len(remote_files) - len(to_download)
    print(f"  ✓ Cache download: {downloaded} downloaded, {skipped} already in sync")
    return (downloaded, skipped)


def main():
    parser = argparse.ArgumentParser(description="Dropbox sync for pipeline CI/CD")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--items", nargs="*", default=[],
                        help="Files to upload/download")
    parser.add_argument("--cache-dir", default=None,
                        help="Persec cache directory to sync (incremental by default)")
    parser.add_argument("--cache-full", action="store_true",
                        help="Use full tar.gz mode instead of incremental sync")
    parser.add_argument("--dropbox-base", default=DROPBOX_BASE,
                        help=f"Dropbox base path (default: {DROPBOX_BASE})")
    parser.add_argument("--local-prefix", default="",
                        help="Local path prefix for items (e.g. athletes/IanLilley)")
    args = parser.parse_args()
    
    token = get_token()
    base = args.dropbox_base
    lp = args.local_prefix.rstrip("/\\") if args.local_prefix else ""
    success = 0
    failed = 0
    
    if args.action == "download":
        print(f"\nDownloading from Dropbox ({base})...")
        
        for item in args.items:
            remote = f"{base}/{item}"
            local = f"{lp}/{item}" if lp else item
            os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
            ok = dropbox_download(remote, local, token)
            if ok:
                success += 1
            else:
                failed += 1
        
        if args.cache_dir:
            if args.cache_full:
                # Full tar.gz download + unpack
                archive = f"{args.cache_dir}.tar.gz"
                remote = f"{base}/cache/{os.path.basename(archive)}"
                if dropbox_download(remote, archive, token):
                    unpack_cache(archive, args.cache_dir)
                    os.remove(archive)  # Clean up local archive
                    success += 1
                else:
                    failed += 1
            else:
                # Incremental: download only missing .npz files
                downloaded, skipped = sync_cache_incremental_download(
                    args.cache_dir, base, token
                )
                if downloaded > 0 or skipped > 0:
                    success += 1
                else:
                    failed += 1
    
    elif args.action == "upload":
        print(f"\nUploading to Dropbox ({base})...")
        
        for item in args.items:
            local = f"{lp}/{item}" if lp else item
            if os.path.exists(local):
                remote = f"{base}/{item}"
                ok = dropbox_upload(local, remote, token)
                if ok:
                    success += 1
                else:
                    failed += 1
        
        if args.cache_dir and os.path.isdir(args.cache_dir):
            if args.cache_full:
                # Full tar.gz pack + upload
                archive = f"{args.cache_dir}.tar.gz"
                if pack_cache(args.cache_dir, archive):
                    remote = f"{base}/cache/{os.path.basename(archive)}"
                    ok = dropbox_upload(archive, remote, token)
                    os.remove(archive)  # Clean up local archive
                    if ok:
                        success += 1
                    else:
                        failed += 1
            else:
                # Incremental: upload only new/changed .npz files
                uploaded, skipped = sync_cache_incremental_upload(
                    args.cache_dir, base, token
                )
                if uploaded > 0 or skipped > 0:
                    success += 1
                # No failure count for "nothing to upload"
    
    print(f"\nDone: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
