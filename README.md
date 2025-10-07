# GeDi: Learning general and distinctive 3D local deep descriptors for point cloud registration - IEEE T-PAMI

Fork of [GeDi](https://github.com/fabiopoiesi/gedi). [Paper (pdf)](https://arxiv.org/pdf/2105.10382.pdf)

## Currently built for:
- Windows/Linux OS
- Python: **3.9/3.10/3.11/3.12**
- CPU
- CUDA: **11.8/12.1/12.4/12.6/12.8/12.9**
- CUDA_ARCH: **6.0/6.1/7.5/8.6/8.9**

## Wheel build for developers
Required: Install [pixi](https://pixi.sh/latest/).

To build the wheel of GeDi, you first need to choose the Python/CPU/CUDA versions with which the wheel will be built (example: **"py312-cu124"**). 
Replace `PIXI_ENV` placeholder in the commands below with your environment name.

Run on Windows:

```
pixi run -e PIXI_ENV wheel-build-windows
```

Run on Linux:

```
pixi run -e PIXI_ENV wheel-build-linux
```

The wheel will be created in `./Release` folder.

## Code overview: registration.py

The registration function `compute_registration_matrix` is structured as follows:

```
def compute_registration_matrix(
    pcd0_points: np.ndarray,
    pcd1_points: np.ndarray,
    samples_per_batch: int = 500,
    samples_per_patch_lrf: int = 4000,
    samples_per_patch_out: int = 512,
    lrf_radius: float = 0.5,
    voxel_size: float = 0.01,
    patches_per_pair: int = 5000,
    max_correspondence_distance: float = 0.02,
    edge_length_checker: float = 0.9,
    distance_checker: float = 0.02,
    ransac_iterations: int = 1000,
    visualize: bool = False,
    device: str = 'cuda'
) -> np.ndarray:
```

**Inputs**

- `pcd0_points` is a NumPy array of shape [N, 3] containing the N-points of the first point cloud;
- `pcd1_points` is a NumPy array of shape [M, 3] containing the M-points of the second point cloud;
- `samples_per_batch` is the number of samples taken from  `patches_per_pair` that are processed in a single batch; the number of batches will therefore be `patches_per_pair/samples_per_batch`;
- `samples_per_patch_lrf` is the number of points to process with LRF;
- `samples_per_patch_out` is the number of points to sample for pointnet++;
- `lrf_radius` is the radius used to compute the Local Reference Frame (LRF) for descriptor computation;
- `voxel_size` is the voxel size used to downsample the point clouds before registration;
- `patches_per_pair` is the number of point patches randomly sampled per point cloud to compute descriptors;
- `max_correspondence_distance` is the maximum distance threshold for matching points during RANSAC;
- `edge_length_checker` is the edge length similarity threshold for the RANSAC correspondence checker;
- `distance_checker` is the distance similarity threshold for the RANSAC correspondence checker;
- `ransac_iterations` is the maximum number of RANSAC iterations;
- `visualize` (bool) whether to visualize the clouds through Open3D visualizer;
- `device` (str) whether to use **CPU** (`arg: 'cpu'`) or **GPU** (`arg: 'cuda'`) to run registration. **Note:** Inference on CPU takes roughly **6× more time** compared to GPU.

**Output**

- A `np.ndarray` 4×4 transformation matrix that aligns `pcd0` to `pcd1`.


The function `compute_registration_matrix_from_path` provides the same registration logic as `compute_registration_matrix`, but loads the point clouds from paths instead of receiving them as NumPy arrays:

**Differences in Inputs**

- `pcd0_path` is the file path (string) to the first point cloud;
- `pcd1_path` is the file path (string) to the second point cloud;

All the other **Inputs** and the **Output** remain the same.

## Use of registration.py from command line

If you want to test or debug locally the registration you can run (replace **PIXI_ENV placeholder** with your environment tag):

```
pixi run -e PIXI_ENV python registration.py --pcd0 /path/to/cloud0 --pcd1 /path/to/cloud1 --visualize --device cuda
```

## Citations

```latex
@inproceedings{Poiesi2021,
  title = {Learning general and distinctive 3D local deep descriptors for point cloud registration},
  author = {Poiesi, Fabio and Boscaini, Davide},
  booktitle = {IEEE Trans. on Pattern Analysis and Machine Intelligence},
  year = {(early access) 2022}
}
```
