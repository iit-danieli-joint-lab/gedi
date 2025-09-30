from pathlib import Path
import shutil
import sys
import platform
import re
import zipfile

# Path to the directory containing the built wheel
dist_dir = Path("dist")

try:
    # ---------------- Step 1: read torch version ----------------
    import torch
    torch_version = torch.__version__
    print(f"Using Torch version: {torch_version}")

    # ---------------- Step 2: determine OS and architecture ----------------
    system = platform.system().lower()  # 'windows', 'linux'
    if system == "windows":
        arch = "win_amd64"
        ext = ".whl"
    elif system == "linux":
        arch = "manylinux_2_28_x86_64"
        ext = ".whl"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    # ---------------- Step 3: determine the PyTorch wheel URL ----------------
    m = re.match(r"(\d+\.\d+\.\d+)(?:\+(\w+))?", torch_version)
    if not m:
        raise RuntimeError(f"Cannot parse torch version: {torch_version}")
    version, cuda_tag = m.groups()
    cuda_tag = cuda_tag or "cpu"

    pyver = f"{sys.version_info.major}{sys.version_info.minor}"  # e.g., 312
    torch_url = f"https://download.pytorch.org/whl/{cuda_tag}/torch-{version}%2B{cuda_tag}-cp{pyver}-cp{pyver}-{arch}{ext}"
    print(f"Using torch URL: {torch_url}")

    # ---------------- Step 4: post-process the wheel ----------------
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError("No wheel found in dist/")
    wheel_path = wheels[0]
    temp_extract = dist_dir / "tmp_wheel"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    temp_extract.mkdir()

    # Extract
    with zipfile.ZipFile(wheel_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract)

    # Find METADATA file
    dist_info_dirs = list(temp_extract.glob("*.dist-info"))
    if not dist_info_dirs:
        raise RuntimeError("No .dist-info folder found in wheel")
    metadata_file = dist_info_dirs[0] / "METADATA"

    # Patch METADATA: replace/add Requires-Dist for torch
    lines = metadata_file.read_text().splitlines()
    new_lines = [l for l in lines if not l.startswith("Requires-Dist: torch")]
    new_lines.append(f"Requires-Dist: torch @ {torch_url}")
    metadata_file.write_text("\n".join(new_lines))

    # Repack wheel (overwrite original)
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_out:
        for file_path in temp_extract.rglob("*"):
            zip_out.write(file_path, file_path.relative_to(temp_extract))

    shutil.rmtree(temp_extract)
    print(f"Wheel patched in-place: {wheel_path}")

except Exception as e:
    print(f"Error occurred: {e}", file=sys.stderr)
    raise
