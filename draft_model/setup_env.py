#!/usr/bin/env python3
"""Create .venv next to this file with PyTorch for this computer's NVIDIA
GPU and transformers; run.bat calls it. Safe to run again: it skips what
is already installed. Everything it runs is also written to setup.log.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
LOG = HERE / "setup.log"
PYTHONS = ["3.12", "3.13", "3.11", "3.10"]  # versions PyTorch has wheels for, preferred first
# transformers 5 cannot read CodeT5's older tokenizer files; 4.57 is the last 4.x
PACKAGES = ["transformers==4.57.6", "numpy"]
BASE = "Salesforce/codet5-small"
CHECK_MODEL = ("from transformers import AutoTokenizer, T5ForConditionalGeneration; "
               f"tok = AutoTokenizer.from_pretrained('{BASE}'); "
               f"model = T5ForConditionalGeneration.from_pretrained('{BASE}'); "
               "print('model ready:', type(tok).__name__, tok('fix a bug', add_special_tokens=False)['input_ids'], "
               "'decoder start', model.config.decoder_start_token_id, 'eos', tok.eos_token_id, "
               "'pad', tok.pad_token_id)")


def log(text):
    print(text, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def run(cmd):
    """Run cmd, showing and logging its output; stop if it fails."""
    log("> " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    lines = []
    for line in proc.stdout:
        lines.append(line)
        log(line.rstrip("\n"))
    if proc.wait():
        raise SystemExit(f"failed with exit code {proc.returncode}; see setup.log")
    return "".join(lines)


def cuda_wheels(nvidia_smi_output):
    """The PyTorch wheel index for what the driver supports: its CUDA
    version, or failing that its driver version. A 12.x driver runs the
    12.6 wheels (CUDA minor version compatibility)."""
    m = re.search(r"CUDA[ A-Za-z]*Version\s*:\s*(\d+)\.(\d+)", nvidia_smi_output)
    if m:
        version = (int(m.group(1)), int(m.group(2)))
    else:
        d = re.search(r"Driver Version\s*:\s*(\d+)\.", nvidia_smi_output)
        if not d:
            return None
        major = int(d.group(1))
        version = (13, 0) if major >= 580 else (12, 8) if major >= 570 else (12, 0) if major >= 525 else (11, 0)
    if version >= (13, 0):
        return "cu130"
    if version >= (12, 8):
        return "cu128"
    if version >= (12, 0):
        return "cu126"
    return None


def pick_python():
    """A Python PyTorch supports: this one, or one the py launcher knows."""
    if f"{sys.version_info.major}.{sys.version_info.minor}" in PYTHONS:
        return sys.executable
    for version in PYTHONS:
        try:
            r = subprocess.run(["py", f"-{version}", "-c", "import sys; print(sys.executable)"],
                               capture_output=True, text=True)
        except FileNotFoundError:
            break
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    raise SystemExit(f"need Python {', '.join(sorted(PYTHONS))}; this is {sys.version.split()[0]}. "
                     "Install Python 3.12 from python.org and run this again.")


def main():
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, errors="replace")
    except FileNotFoundError:
        raise SystemExit("nvidia-smi not found: install or update the NVIDIA driver first")
    output = (smi.stdout or "") + (smi.stderr or "")
    log(output.rstrip())
    if smi.returncode:
        raise SystemExit("nvidia-smi failed (above): the NVIDIA driver is not working")
    index = cuda_wheels(output)
    if not index:
        raise SystemExit("the NVIDIA driver is too old for current PyTorch (needs CUDA 12 or later); "
                         "update it and run this again")
    log(f"NVIDIA driver supports {index}")
    if not PY.exists():
        run([pick_python(), "-m", "venv", VENV])
    run([PY, "-m", "pip", "install", "--upgrade", "pip"])
    ready = subprocess.run([PY, "-c", "import torch; print(torch.cuda.is_available())"],
                           capture_output=True, text=True).stdout.strip()
    if ready != "True":
        run([PY, "-m", "pip", "install", "--upgrade", "torch", "--index-url",
             f"https://download.pytorch.org/whl/{index}"])
    run([PY, "-m", "pip", "install", *PACKAGES])
    check = run([PY, "-c", "import torch, transformers; print('torch', torch.__version__, 'transformers', "
                           "transformers.__version__, 'GPU', torch.cuda.is_available() and "
                           "torch.cuda.get_device_name(0))"])
    if "GPU False" in check:
        raise SystemExit("PyTorch cannot see the GPU; see setup.log")
    log(f"loading {BASE} (the first time downloads about 240 MB)")
    run([PY, "-c", CHECK_MODEL])
    log("environment ready")


if __name__ == "__main__":
    main()
