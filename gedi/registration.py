import sys
import torch
import numpy as np
import open3d as o3d

def compute_registration_matrix_from_path(
    pcd0_path: str,
    pcd1_path: str,
    lrf_radius: float = 0.5,
    voxel_size: float = 0.01,
    patches_per_pair: int = 5000,
    max_correspondence_distance: float = 0.02,
    edge_length_checker: float = 0.9,
    distance_checker: float = 0.02,
    ransac_iterations: int = 1000,
    visualize: bool = False
) -> np.ndarray:
    pcd0 = o3d.io.read_point_cloud(pcd0_path)
    pcd1 = o3d.io.read_point_cloud(pcd1_path)

    return compute_registration_matrix(    
    np.asarray(pcd0.points),
    np.asarray(pcd1.points),
    lrf_radius,
    voxel_size,
    patches_per_pair,
    max_correspondence_distance,
    edge_length_checker,
    distance_checker,
    ransac_iterations,
    visualize) 

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
    visualize: bool = False
) -> np.ndarray:
    from .gedi import GeDi

    try:
        import importlib.resources as pkg_resources
    except ImportError:
        import importlib_resources as pkg_resources  

    # getting checkpoint
    with pkg_resources.files('gedi').joinpath('assets/chkpt.tar') as chkpt_path:
        fchkpt_gedi_net = str(chkpt_path)

    config = {
        'dim': 32,                              # descriptor output dimension
        'samples_per_batch': 500,               # batches to process the data on GPU
        'samples_per_patch_lrf': 4000,          # num. of point to process with LRF
        'samples_per_patch_out': 512,           # num. of points to sample for pointnet++
        'r_lrf': lrf_radius,                    # LRF radius from input
        'fchkpt_gedi_net': fchkpt_gedi_net      # path to checkpoint
    }

    # load point clouds
    pcd0 = o3d.geometry.PointCloud()
    pcd1 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pcd0_points)
    pcd1.points = o3d.utility.Vector3dVector(pcd1_points)

    # initialise GeDi
    gedi = GeDi(config=config)

    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])

    # estimate normals 
    pcd0.estimate_normals()
    pcd1.estimate_normals()

    if visualize:
        o3d.visualization.draw_geometries([pcd0, pcd1])

    # randomly sample points
    inds0 = np.random.choice(len(pcd0.points), patches_per_pair, replace=False)
    inds1 = np.random.choice(len(pcd1.points), patches_per_pair, replace=False)

    pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
    pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

    # voxel downsample
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)

    _pcd0 = torch.tensor(np.asarray(pcd0_down.points)).float()
    _pcd1 = torch.tensor(np.asarray(pcd1_down.points)).float()

    # compute descriptors
    pcd0_desc = gedi.compute(pts=pts0, pcd=_pcd0)
    pcd1_desc = gedi.compute(pts=pts1, pcd=_pcd1)

    # prepare for Open3D RANSAC
    pcd0_dsdv = o3d.pipelines.registration.Feature()
    pcd1_dsdv = o3d.pipelines.registration.Feature()
    pcd0_dsdv.data = pcd0_desc.T
    pcd1_dsdv.data = pcd1_desc.T

    print("Number of descriptors in pcd0:", pcd0_dsdv.data.shape[1])
    print("Number of descriptors in pcd1:", pcd1_dsdv.data.shape[1])

    _pcd0_o3d = o3d.geometry.PointCloud()
    _pcd0_o3d.points = o3d.utility.Vector3dVector(pts0)
    _pcd1_o3d = o3d.geometry.PointCloud()
    _pcd1_o3d.points = o3d.utility.Vector3dVector(pts1)

    # run RANSAC registration
    est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        _pcd0_o3d,
        _pcd1_o3d,
        pcd0_dsdv,
        pcd1_dsdv,
        mutual_filter=True,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(edge_length_checker),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_checker)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, ransac_iterations)
    )

    print(f"Number of correspondences found by RANSAC: {len(est_result01.correspondence_set)}")

    if visualize:
        pcd0.transform(est_result01.transformation)
        o3d.visualization.draw_geometries([pcd0, pcd1])

    return est_result01.transformation
