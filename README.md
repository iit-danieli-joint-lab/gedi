# GeDi: Learning general and distinctive 3D local deep descriptors for point cloud registration - IEEE T-PAMI

Fork of [GeDi](https://github.com/fabiopoiesi/gedi). [Paper (pdf)](https://arxiv.org/pdf/2105.10382.pdf)

## Tested with

### Linux 
- Ubuntu 22.04
- CUDA 12.1 (optional)
- Python 3.12
- Torch 2.2.2
- Open3D 0.19.0
- [torchgeometry v0.1.2](https://kornia.readthedocs.io/en/v0.1.2/)

### Windows

- Windows 11
- CUDA 12.8 (optional)
- Python 3.12
- Torch 2.7.1
- Open3D 0.19.0+083210b
- [torchgeometry v0.1.2](https://kornia.readthedocs.io/en/v0.1.2/)

## Installation

**(Optional)** If you want to update the version of `pointnet2_ops` see [pointnet2_ops Installation](#pointnet2_ops-installation).

To build the wheel of GeDi:

```
pixi run build-gedi
```

The wheel will be created in `./dist` folder.

## pointnet2_ops Installation

GeDi needs `pointnet2_ops`, so you need to create a wheel and then upload it in the latest release (see [Releases](https://github.com/iit-danieli-joint-lab/gedi/releases/)).

Go to:

```
cd gedi/backbones/pointnet2_ops_lib
```

and in `pyproject.toml` ensure you have (un)commented the correct `torch` version for your setup (CPU or CUDA). Then to build the wheel:

```
pixi run build-pointnet2_ops
```

The wheel will be created in `./dist` folder.

## registration.py

The registration function `compute_registration_matrix` is structured as follows:

```
def compute_registration_matrix(
    pcd0_points: np.ndarray,
    pcd1_points: np.ndarray,
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

## Citations

```latex
@inproceedings{Poiesi2021,
  title = {Learning general and distinctive 3D local deep descriptors for point cloud registration},
  author = {Poiesi, Fabio and Boscaini, Davide},
  booktitle = {IEEE Trans. on Pattern Analysis and Machine Intelligence},
  year = {(early access) 2022}
}
```
