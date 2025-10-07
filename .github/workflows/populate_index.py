import os
import shutil

# --- Configuration ---
base_dir = "whl-index"
subdirs = ["cpu", "cu118", "cu121", "cu124", "cu126", "cu128", "cu129"]
release_dir = "Release"  # folder where wheels are downloaded

# --- Create base structure if it does not exist ---
os.makedirs(base_dir, exist_ok=True)
index_main = os.path.join(base_dir, "index.html")
if not os.path.exists(index_main):
    with open(index_main, "w") as f:
        f.write("<html><body>\n</body></html>")

for d in subdirs:
    subdir_path = os.path.join(base_dir, d)
    os.makedirs(subdir_path, exist_ok=True)
    index_file = os.path.join(subdir_path, "index.html")
    if not os.path.exists(index_file):
        with open(index_file, "w") as f:
            f.write("<html><body>\n</body></html>")

# --- Function to update index without duplicates ---
def update_index(wheel_path):
    whl_name = os.path.basename(wheel_path)
    if "+cpu" in whl_name:
        subdir = os.path.join(base_dir, "cpu")
    elif "+cu" in whl_name:
        cu_ver = whl_name.split("+cu")[1].split("_")[0]
        subdir = os.path.join(base_dir, f"cu{cu_ver}")
    else:
        subdir = base_dir

    os.makedirs(subdir, exist_ok=True)
    dest_path = os.path.join(subdir, whl_name)
    if not os.path.exists(dest_path):
        shutil.copy2(wheel_path, dest_path)

    index_file = os.path.join(subdir, "index.html")
    with open(index_file, "r") as f:
        content = f.read()
    if whl_name not in content:
        content = content.replace("</body></html>", f'<a href="{whl_name}">{whl_name}</a><br>\n</body></html>')
        with open(index_file, "w") as f:
            f.write(content)

# --- Update all wheels in the Release folder ---
if os.path.exists(release_dir):
    for whl in os.listdir(release_dir):
        if whl.endswith(".whl"):
            update_index(os.path.join(release_dir, whl))
else:
    print(f"Directory '{release_dir}' not found. No wheels to process.")
