import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse, unquote

report_path = Path("report.json")
wheels_dir = Path("wheels")
wheels_dir.mkdir(exist_ok=True)

with open(report_path, "r", encoding="utf-8") as f:
    data = json.load(f)

install_items = data.get("install", [])
urls = []
for item in install_items:
    d = item.get("download_info", {})
    url = d.get("url")
    if url:
        urls.append(url)

print(f"Total wheels to download: {len(urls)}")

for i, url in enumerate(urls, 1):
    parsed = urlparse(url)
    filename = Path(unquote(parsed.path)).name
    target = wheels_dir / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"[{i}/{len(urls)}] Skipping already downloaded: {filename}")
        continue

    print(f"[{i}/{len(urls)}] Downloading: {filename}")
    cmd = ["curl.exe", "-L", "-C", "-", "-o", str(target), url]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Warning: curl returned {res.returncode} for {filename}")

print("All wheel downloads completed.")
