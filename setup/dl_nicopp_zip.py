"""Download NICO++ as a single zip from 2018LZY/NICOplusplus and unpack."""
import os, sys, zipfile
from pathlib import Path

for v in ("HF_HOME", "HF_HUB_CACHE"):
    if v not in os.environ: sys.exit(f"set {v}")
TOKEN = open("/coralation-analisis/.token").read().strip()
os.environ.setdefault("HF_TOKEN", TOKEN)

from huggingface_hub import hf_hub_download

OUT = Path("/root/data/nicopp")
OUT.mkdir(parents=True, exist_ok=True)

print("[NICOpp.zip] downloading...", flush=True)
zip_path = hf_hub_download(
    repo_id="2018LZY/NICOplusplus", filename="NICOpp.zip",
    repo_type="dataset", token=TOKEN,
)
print(f"[NICOpp.zip] OK: {zip_path}, size={Path(zip_path).stat().st_size/1024/1024:.1f} MB", flush=True)

print(f"[NICOpp.zip] unpacking to {OUT}...", flush=True)
with zipfile.ZipFile(zip_path) as zf:
    members = zf.namelist()
    print(f"  {len(members)} entries; sample top-levels: {sorted(set(m.split('/')[0] for m in members))[:10]}")
    zf.extractall(OUT)
print(f"[NICOpp.zip] unpacked. Layout under {OUT}/:")
import subprocess
subprocess.run(["find", str(OUT), "-maxdepth", "3", "-type", "d"], check=False)
