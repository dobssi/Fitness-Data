"""
check_strava_zip.py — Drop this next to your Strava export zip and run it.

Usage:
    python check_strava_zip.py export_12345678.zip
    
Or just double-click — it will find any zip in the same folder.
"""
import zipfile
import collections
import glob
import os
import sys

# Find the zip
if len(sys.argv) > 1:
    zip_path = sys.argv[1]
else:
    # Look for a zip in the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zips = glob.glob(os.path.join(script_dir, "*.zip"))
    if not zips:
        print("No zip file found. Place this script next to your Strava export zip.")
        input("Press Enter to exit...")
        sys.exit(1)
    zip_path = zips[0]
    print(f"Found: {os.path.basename(zip_path)}")

print(f"Scanning: {zip_path}\n")

z = zipfile.ZipFile(zip_path)
exts = collections.Counter()
has_activities_csv = False

for f in z.namelist():
    if f.lower().endswith("activities.csv"):
        has_activities_csv = True
    if "activities/" in f.lower() and not f.endswith("/"):
        if f.endswith(".gz"):
            parts = f.rsplit(".", 2)
            ext = parts[-2] + ".gz" if len(parts) >= 3 else "gz"
        else:
            ext = f.rsplit(".", 1)[-1]
        exts[ext.lower()] += 1

print("Activity files by type:")
for ext, n in exts.most_common():
    print(f"  .{ext}: {n}")
print(f"\nTotal activity files: {sum(exts.values())}")
print(f"activities.csv found: {'Yes' if has_activities_csv else 'No'}")

input("\nPress Enter to exit...")
