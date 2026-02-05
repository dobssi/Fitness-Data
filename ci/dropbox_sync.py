"""
ci/dropbox_sync.py — Upload/download pipeline files to/from Dropbox
===================================================================
Used by GitHub Actions to persist pipeline state between runs.

Usage:
  python ci/dropbox_sync.py download --items file1.xlsx file2.csv
  python ci/dropbox_sync.py upload --items file1.xlsx file2.csv
  python ci/dropbox_sync.py download --cache-dir persec_cache_FULL

Environment variables:
  DROPBOX_TOKEN — OAuth2 access token with files.content.write scope
  
Dropbox folder layout:
  /Running and Cycling/DataPipeline/
    ├── Master_FULL_GPSQ_ID.xlsx
    ├── Master_FULL_GPSQ_ID_post.xlsx
    ├── athlete_data.csv
    ├── re_model_s4_FULL.json
    ├── sync_state.json
    ├── TotalHistory.zip
    └── cache/
        └── persec_cache_FULL.tar.gz
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

# Dropbox upload limit per request: 150 MB
# For larger files, use upload_session — but most pipeline files are <50 MB
MAX_UPLOAD_SIZE = 150 * 1024 * 1024


def get_token() -> str:
    token = os.environ.get("DROPBOX_TOKEN", "")
    if not token:
        sys.exit("ERROR: DROPBOX_TOKEN environment variable not set")
    return token


def dropbox_upload(local_path: str, remote_path: str, token: str):
    """Upload a single file to Dropbox."""
    if not os.path.exists(local_path):
        print(f"  ⚠ Skip (not found): {local_path}")
        return False
    
    size = os.path.getsize(local_path)
    if size > MAX_UPLOAD_SIZE:
        print(f"  ⚠ Skip (too large: {size / 1024 / 1024:.0f} MB): {local_path}")
        return False
    
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


def main():
    parser = argparse.ArgumentParser(description="Dropbox sync for pipeline CI/CD")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--items", nargs="*", default=[],
                        help="Files to upload/download")
    parser.add_argument("--cache-dir", default=None,
                        help="Persec cache directory to sync")
    parser.add_argument("--dropbox-base", default=DROPBOX_BASE,
                        help=f"Dropbox base path (default: {DROPBOX_BASE})")
    args = parser.parse_args()
    
    token = get_token()
    base = args.dropbox_base
    success = 0
    failed = 0
    
    if args.action == "download":
        print(f"\nDownloading from Dropbox ({base})...")
        
        for item in args.items:
            remote = f"{base}/{item}"
            ok = dropbox_download(remote, item, token)
            if ok:
                success += 1
            else:
                failed += 1
        
        if args.cache_dir:
            archive = f"{args.cache_dir}.tar.gz"
            remote = f"{base}/cache/{os.path.basename(archive)}"
            if dropbox_download(remote, archive, token):
                unpack_cache(archive, args.cache_dir)
                os.remove(archive)  # Clean up local archive
                success += 1
            else:
                failed += 1
    
    elif args.action == "upload":
        print(f"\nUploading to Dropbox ({base})...")
        
        for item in args.items:
            if os.path.exists(item):
                remote = f"{base}/{item}"
                ok = dropbox_upload(item, remote, token)
                if ok:
                    success += 1
                else:
                    failed += 1
        
        if args.cache_dir and os.path.isdir(args.cache_dir):
            archive = f"{args.cache_dir}.tar.gz"
            if pack_cache(args.cache_dir, archive):
                remote = f"{base}/cache/{os.path.basename(archive)}"
                ok = dropbox_upload(archive, remote, token)
                os.remove(archive)  # Clean up local archive
                if ok:
                    success += 1
                else:
                    failed += 1
    
    print(f"\nDone: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
