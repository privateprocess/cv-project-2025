import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import plotly.graph_objects as go
import os
from pathlib import Path
from collections import defaultdict 


CAMERA_NAMES = [f'C{i}' for i in range(1, 8)]

# Base path
calib_path = Path("data/Wildtrack/calibrations")

# Load extrinsics
extrinsics = {}
extr_files = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]

for name in extr_files:
    fs = cv.FileStorage(str(calib_path / "extrinsic" / f"extr_{name}.xml"), cv.FILE_STORAGE_READ)
    rvec_node = fs.getNode('rvec')
    tvec_node = fs.getNode('tvec')
    
    extrinsics[name] = {
        'rvec': np.array([rvec_node.at(i).real() for i in range(int(rvec_node.size()))]),
        'tvec': np.array([tvec_node.at(i).real() for i in range(int(tvec_node.size()))])
    }
    fs.release()

# Load intrinsics
intrinsics = {}
intr_files = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]

for name in intr_files:
    fs = cv.FileStorage(str(calib_path / "intrinsic_zero" / f"intr_{name}.xml"), cv.FILE_STORAGE_READ) # Our images are already undistorted; use intrinsic_zero
    intrinsics[name] = {
        'camera_matrix': fs.getNode('camera_matrix').mat(),
        'dist_coeffs': fs.getNode('distortion_coefficients').mat().flatten()
    }
    fs.release()


############################
# Visualization functions
############################
def add_camera_visuals(
    fig,
    camera_center,
    camera_name,
    extrinsics,
    scale=200,
    arrow_size=100
):
    C = np.asarray(camera_center).reshape(3)

    # ---- Camera center ----
    fig.add_trace(
        go.Scatter3d(
            x=[C[0]], y=[C[1]], z=[C[2]],
            mode='markers+text',
            marker=dict(size=5, color='red'),
            text=[camera_name],
            textposition='top center',
            name=f'Camera {camera_name} Center'
        )
    )

    # ---- Extrinsics ----
    filename = convert_camera_name_to_filename(camera_name)

    rvec = np.asarray(extrinsics[filename]['rvec']).reshape(3)
    R, _ = cv.Rodrigues(rvec)

    # OpenCV: world -> camera
    # We need camera -> world
    R_world = R.T

    # ---- Camera forward direction (+Z in camera frame) ----
    ray_camera = np.array([0.0, 0.0, 1.0])

    camera_direction = R_world @ ray_camera
    camera_direction /= np.linalg.norm(camera_direction)

    end_point = C + camera_direction * scale

    # ---- Direction line ----
    fig.add_trace(
        go.Scatter3d(
            x=[C[0], end_point[0]],
            y=[C[1], end_point[1]],
            z=[C[2], end_point[2]],
            mode='lines',
            line=dict(width=3),
            name=f'Camera {camera_name} Direction'
        )
    )

    # ---- Arrowhead ----
    fig.add_trace(
        go.Cone(
            x=[end_point[0]],
            y=[end_point[1]],
            z=[end_point[2]],
            u=[camera_direction[0]],
            v=[camera_direction[1]],
            w=[camera_direction[2]],
            sizemode='absolute',
            sizeref=arrow_size,
            showscale=False,
            colorscale=[[0, 'blue'], [1, 'blue']],
            name=f'Camera {camera_name} Arrow',
            showlegend=False
        )
    )


def visualize_predictions(
    image,
    bboxes,
    scores,
    categories,
    score_threshold=0.5,
    figsize=(16, 12),
    show_labels=True,
    show_scores=True,
    show_indices=True,
    font_size=10,
    box_thickness=2
):
    """
    Visualize bounding box predictions on an image.
    
    Args:
        image: Input image (numpy array, BGR or RGB)
        bboxes: List of bounding boxes in xyxy format [[x1, y1, x2, y2], ...]
        scores: List of confidence scores
        categories: List of category names
        score_threshold: Minimum score to display (default: 0.5)
        figsize: Figure size (default: (16, 12))
        show_labels: Whether to show category labels (default: True)
        show_scores: Whether to show confidence scores (default: True)
        show_indices: Whether to show bounding box indices (default: True)
        font_size: Font size for labels (default: 10)
        box_thickness: Thickness of bounding box lines (default: 2)
    """
    # Convert BGR to RGB if needed (OpenCV uses BGR)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume it might be BGR, convert to RGB for display
        display_image = image.copy()
        if display_image.max() > 1:
            display_image = display_image.astype(np.uint8)
    else:
        display_image = image
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(
        cv.cvtColor(display_image, cv.COLOR_BGR2RGB)
    )
    
    # Generate unique colors for each category
    unique_categories = list(set(categories))
    category_colors = {}
    cmap = plt.get_cmap('tab10')
    for idx, cat in enumerate(unique_categories):
        category_colors[cat] = cmap(idx % 10)
    
    # Filter by score threshold
    filtered_indices = [i for i, score in enumerate(scores) if score >= score_threshold]
    
    # Draw bounding boxes
    for idx in filtered_indices:
        bbox = bboxes[idx]
        score = scores[idx]
        category = categories[idx]
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Get color for this category
        color = category_colors[category]
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), 
            width, 
            height,
            linewidth=box_thickness,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with background
        if show_labels or show_scores or show_indices:
            label_parts = []
            if show_indices:
                label_parts.append(f'[{idx}]')
            if show_labels:
                label_parts.append(category)
            if show_scores:
                label_parts.append(f'{score:.2f}')
            label = ' '.join(label_parts)
            
            # Add text with background box
            ax.text(
                x1, 
                y1 - 5,
                label,
                fontsize=font_size,
                color='white',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=color,
                    alpha=0.7,
                    edgecolor='none'
                ),
                verticalalignment='bottom'
            )
    
    # Add legend with category counts
    legend_elements = []
    for cat in unique_categories:
        count = sum(1 for i in filtered_indices if categories[i] == cat)
        legend_elements.append(
            patches.Patch(
                facecolor=category_colors[cat],
                label=f'{cat} ({count})'
            )
        )
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=font_size,
        framealpha=0.8
    )
    
    ax.set_title(
        f'Object Detection Results (Threshold: {score_threshold})\n'
        f'Total Detections: {len(filtered_indices)}',
        fontsize=font_size + 4,
        pad=20
    )
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"Detection Summary")
    print(f"{'='*50}")
    print(f"Total detections: {len(bboxes)}")
    print(f"Detections above threshold ({score_threshold}): {len(filtered_indices)}")
    print(f"\nCategory breakdown:")
    for cat in unique_categories:
        count = sum(1 for i in filtered_indices if categories[i] == cat)
        avg_score = np.mean([scores[i] for i in filtered_indices if categories[i] == cat])
        print(f"  {cat}: {count} (avg score: {avg_score:.3f})")
    print(f"{'='*50}\n")
    
# display list of images with image channel handling
def display_images(*imgs):
    n = len(imgs)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        if len(imgs[i].shape) == 2:  # grayscale
            plt.imshow(imgs[i], cmap='gray')
        else:  # color
            plt.imshow(cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

def load_wildtrack_images(image_dir, num_images=None):
    folders = [f"C{i}" for i in range(1, 8)]

    # Use the first folder to define frame list
    base_dir = os.path.join(image_dir, folders[0])

    files = [
        f for f in os.listdir(base_dir)
        if f.endswith(".png") and len(os.path.splitext(f)[0]) == 8
    ]

    # Sort by numeric frame id
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    selected_files = files[:num_images]

    rows = []

    for fname in tqdm(selected_files):
        frame_num = int(os.path.splitext(fname)[0])
        row = {}

        # Load images from each folder
        for folder in folders:
            img_path = os.path.join(image_dir, folder, fname)
            if os.path.exists(img_path):
                row[folder] = cv.imread(img_path)
            else:
                row[folder] = None

        # Store the real frame id last
        row["frame_id"] = frame_num
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index = range(len(df))
    return df

def create_camera_animation(frames, save_path, folder_path="saved_files", fps=30):
    num_frames = len(frames)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Initialize with first frame
    im = ax.imshow(
        cv.cvtColor(frames[0], cv.COLOR_BGR2RGB)
    )
    title = ax.set_title(f'Frame 1/{num_frames}', fontsize=14, fontweight='bold')
    
    def update(frame_idx):
        im.set_array(
            cv.cvtColor(frames[frame_idx], cv.COLOR_BGR2RGB)
        )
        title.set_text(f'Frame {frame_idx + 1}/{num_frames}')
        return [im, title]
    
    # Create animation
    anim = FuncAnimation(
        fig, update,
        frames=num_frames,
        interval=1000/fps,
        blit=True,
        repeat=True
    )
    
    # Save animation
    if save_path.endswith('.gif'):
        anim.save(f"{folder_path}/{save_path}", writer='pillow', fps=fps)
    else:
        anim.save(f"{folder_path}/{save_path}", writer='ffmpeg', fps=fps, dpi=100)
    
    plt.close()
    print(f'Animation saved to {folder_path}/{save_path}')
    
    return anim

############################
# Wedge functions
############################

def convert_camera_name_to_filename(camera_name: str) -> str:
    """Convert camera name to corresponding filename index."""
    calibration_filenames = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]

    conversion_dict = dict(zip(CAMERA_NAMES, calibration_filenames))
    return conversion_dict.get(camera_name, None)

def image_points_to_rays(points, camera_name):

    """Convert multiple 2D image points to 3D rays in world coordinates."""
    # load intrinsics and extrinsics; no need for distortion correction as points are already undistorted
    filename = convert_camera_name_to_filename(camera_name)

    K = intrinsics[filename]['camera_matrix']
    rvec = extrinsics[filename]['rvec']
    tvec = extrinsics[filename]['tvec']

    # Convert rotation vector to matrix
    R, _ = cv.Rodrigues(rvec)

    # homogeneous coordinates
    p = np.insert(points, 2, 1.0, axis=1)

    # Convert to direction ray (from camera center) using the formal "X = (R'*K^-1*p)/s - R'*T" --> X is a point ray from C=-R'*T along direction "R' * [undistorted point]"
    K_inv = np.linalg.inv(K)

    rays = []
    for i in range(points.shape[0]):
        ray_dir_world = R.T @ (K_inv @ p[i])
        rays.append(ray_dir_world / np.linalg.norm(ray_dir_world))
    rays = np.array(rays)

    # Camera center in world coordinates (ray origin)
    C = -R.T @ tvec

    return C, rays

def world_points_to_image_points(points_3d, camera_name):
    """
    Project multiple 3D world points to 2D image coordinates.
    
    Args:
        points_3d: Nx3 array of 3D points in world coordinates
        camera_name: Name of the camera to project to
        
    Returns:
        points_2d: Nx2 array of 2D image coordinates
    """
    # Load intrinsics and extrinsics
    filename = convert_camera_name_to_filename(camera_name)
    K = intrinsics[filename]['camera_matrix']
    rvec = extrinsics[filename]['rvec']
    tvec = extrinsics[filename]['tvec']
    
    # Convert rotation vector to matrix
    R, _ = cv.Rodrigues(rvec)
    
    # Ensure points_3d is a numpy array
    points_3d = np.array(points_3d)
    
    # Convert 3D points to homogeneous coordinates (Nx4)
    points_3d_homogeneous = np.insert(points_3d, 3, 1.0, axis=1)
    
    # Create the extrinsic matrix [R | T] (3x4)
    RT = np.hstack([R, tvec])
    
    # Project: p = K * [R | T] * P
    # This gives us homogeneous 2D coordinates (3xN)
    points_2d_homogeneous = K @ RT @ points_3d_homogeneous.T
    
    # Convert from homogeneous to cartesian coordinates by dividing by the third coordinate (s)
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]
    
    # Transpose to get Nx2 array
    points_2d = points_2d.T
    
    return points_2d

def world_points_to_image_points(points_3d, camera_name):
    """
    Project multiple 3D world points to 2D image coordinates.
    
    Args:
        points_3d: Nx3 array of 3D points in world coordinates
        camera_name: Name of the camera to project to
        
    Returns:
        points_2d: Nx2 array of 2D image coordinates
    """
    # Load intrinsics and extrinsics
    filename = convert_camera_name_to_filename(camera_name)
    K = intrinsics[filename]['camera_matrix']
    rvec = extrinsics[filename]['rvec']
    tvec = extrinsics[filename]['tvec']
    
    # Convert rotation vector to matrix
    R, _ = cv.Rodrigues(rvec)
    
    # Ensure points_3d is a numpy array
    points_3d = np.array(points_3d)
    
    # Convert 3D points to homogeneous coordinates (Nx4)
    points_3d_homogeneous = np.insert(points_3d, 3, 1.0, axis=1)
    
    # Create the extrinsic matrix [R | T] (3x4)
    tvec = tvec.reshape(-1,1)
    RT = np.hstack([R, tvec])
    
    # Project: p = K * [R | T] * P
    # This gives us homogeneous 2D coordinates (3xN)
    points_2d_homogeneous = K @ RT @ points_3d_homogeneous.T
    
    # Convert from homogeneous to cartesian coordinates by dividing by the third coordinate (s)
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]
    
    # Transpose to get Nx2 array
    points_2d = points_2d.T
    
    return points_2d

def quad_area(rays):
    """Fast area calculation for planar 3D quadrilateral using diagonals."""
    d1 = rays[2] - rays[0]
    d2 = rays[3] - rays[1]
    return 0.5 * np.linalg.norm(np.cross(d1, d2))


############################
# Wedge intersection / object mapping functions
############################

# OPTIMIZATION 1: Generate voxel points as numpy array for faster operations
def generate_voxel_grid(xrange, yrange, zrange, spacing):
    """Generate voxel grid as numpy array for vectorized operations"""
    x = np.arange(xrange[0], xrange[1], spacing)
    y = np.arange(yrange[0], yrange[1], spacing)
    z = np.arange(zrange[0], zrange[1], spacing)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    return points

# OPTIMIZATION 2: Fully vectorized wedge checking for batches of points
def point_in_wedge_vectorized(points, origin, rays):
    """
    Vectorized version that checks multiple points at once.
    
    Args:
        points: (N, 3) array of 3D points to check
        origin: (3,) camera origin
        rays: (4, 3) array of ray directions defining the wedge (assumes 4 rays)
        
    Returns:
        in_wedge: (N,) boolean array indicating if each point is in wedge
        scores: (N,) float array with alignment scores
    """
    # Compute vectors from origin to all points: (N, 3)
    v = points - origin
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms > 1e-10, norms, 1e-10)
    v = v / norms  # (N, 3) normalized vectors
    
    # Compute centroid of rays (approximate center of wedge): (3,)
    centroid = np.mean(rays, axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    
    n_points = points.shape[0]
    n_rays = rays.shape[0]
    
    # Initialize results
    in_wedge = np.ones(n_points, dtype=bool)
    
    # Process each edge of the wedge
    for i in range(n_rays):
        ray1 = rays[i]  # (3,)
        ray2 = rays[(i + 1) % n_rays]  # (3,)
        
        # Compute normal for this edge: (3,)
        normal = np.cross(ray1, ray2)
        
        # Determine if normal points inward or outward using centroid
        centroid_side = np.dot(centroid, normal)
        
        # Compute dot product for all points with this normal: (N,)
        v_dot_normal = np.dot(v, normal)
        
        if centroid_side > 0:
            # Points must be on positive side (same as centroid)
            in_wedge &= (v_dot_normal >= 0)
        else:
            # Points must be on negative side (same as centroid)
            in_wedge &= (v_dot_normal <= 0)    
    
    return in_wedge

# OPTIMIZATION 3: Process each camera in parallel
def process_camera(cam_name, voxel_points, camera_position, ray_slice_data, 
                   ray_slice_area_data, category_data, score_data, category_to_track):
    """Process all voxels for a single camera"""
    
    ray_slices = ray_slice_data[cam_name]
    slice_areas = ray_slice_area_data[cam_name]
    categories = category_data[cam_name]
    slice_scores = score_data[cam_name]
    
    # Pre-filter slices by category
    valid_slices = [i for i, cat in enumerate(categories) 
                    if cat.lower() == category_to_track.lower()]
    
    if not valid_slices:
        return cam_name, {}
    
    # Initialize result storage for this camera
    cam_results = {}
    
    # Calculate distances once for all voxels (vectorized)
    distances_squared = np.sum((voxel_points - camera_position) ** 2, axis=1)
    
    # Process each valid slice
    for slice_idx in valid_slices:
        rays = ray_slices[slice_idx]
        area = slice_areas[slice_idx]
        score = slice_scores[slice_idx]
        
        # Vectorized wedge check for all points at once
        in_wedge_mask = point_in_wedge_vectorized(
            points=voxel_points,
            origin=camera_position,
            rays=rays
        )
        
        # Calculate weights for points in wedge
        weights = score / (area * distances_squared)
        
        # Store results for points in wedge
        for voxel_idx in np.where(in_wedge_mask)[0]:
            if voxel_idx not in cam_results:
                cam_results[voxel_idx] = {
                    'slice_idx': [],
                    'slice_weight': []
                }
            cam_results[voxel_idx]['slice_idx'].append(slice_idx)
            cam_results[voxel_idx]['slice_weight'].append(weights[voxel_idx])
    
    return cam_name, cam_results

# MAIN PROCESSING with multithreading
def process_all_cameras_parallel(voxel_points, CAMERA_NAMES, camera_positions,
                                 ray_slice_data, ray_slice_area_data, 
                                 category_data, score_data, category_to_track,
                                 max_workers=None):
    """
    Process all cameras in parallel using ThreadPoolExecutor.
    
    Args:
        voxel_points: (N, 3) numpy array of voxel positions
        CAMERA_NAMES: List of camera names
        camera_positions: Dict mapping camera names to (3,) position arrays
        ray_slice_data: Dict mapping camera names to ray slice data
        ray_slice_area_data: Dict mapping camera names to slice areas
        category_data: Dict mapping camera names to category lists
        score_data: Dict mapping camera names to score lists
        category_to_track: String category to filter
        max_workers: Number of threads (default: number of cameras, capped at 8)
    
    Returns:
        voxel_dict: Dict mapping voxel indices to camera data
        valid_point_indices: List of voxel indices with detections
    """
    if max_workers is None:
        max_workers = min(len(CAMERA_NAMES), 8)  # Cap at 8 threads
    
    # Initialize results structure
    voxel_dict = {}
    points_in_any_slice = set()
    
    print(f"Processing {len(CAMERA_NAMES)} cameras with {max_workers} threads...")
    print(f"Total voxels: {len(voxel_points)}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all camera processing tasks
        future_to_cam = {
            executor.submit(
                process_camera,
                cam,
                voxel_points,
                camera_positions[cam],
                ray_slice_data,
                ray_slice_area_data,
                category_data,
                score_data,
                category_to_track
            ): cam for cam in CAMERA_NAMES
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_cam), total=len(CAMERA_NAMES), 
                          desc="Processing cameras"):
            cam_name, cam_results = future.result()
            
            # Merge results
            for voxel_idx, data in cam_results.items():
                if voxel_idx not in voxel_dict:
                    voxel_dict[voxel_idx] = {}
                voxel_dict[voxel_idx][cam_name] = data
                points_in_any_slice.add(voxel_idx)
    
    # Filter to only keep active voxels
    active_voxels = {idx: voxel_dict[idx] for idx in points_in_any_slice}
    
    print(f"Kept {len(active_voxels)}/{len(voxel_points)} voxels with detections")
    
    return active_voxels, list(points_in_any_slice)

# UTILITY: Extract camera positions from extrinsics
def get_camera_positions(extrinsics, CAMERA_NAMES, convert_camera_name_to_filename):
    """
    Extract camera positions from extrinsics using Rodrigues rotation vectors.
    
    Camera position C = -R^-1 * T = -R' * T
    where R is computed from rvec using Rodrigues formula
    
    Args:
        extrinsics: Dict mapping filename keys to extrinsic parameters
                   Format: {filename: {'rvec': array([...]), 'tvec': array([...])}}
        CAMERA_NAMES: List of camera names to process
        convert_camera_name_to_filename: Function that converts camera name to filename key
    
    Returns:
        camera_positions: Dict mapping camera names to (3,) position arrays
    """
    
    camera_positions = {}
    
    for cam_name in CAMERA_NAMES:
        # Convert camera name to filename key
        filename_key = convert_camera_name_to_filename(cam_name)
        
        if filename_key not in extrinsics:
            raise ValueError(f"Camera {cam_name} (filename: {filename_key}) not found in extrinsics")
        
        # Extract rvec and tvec
        rvec = np.array(extrinsics[filename_key]['rvec'])
        tvec = np.array(extrinsics[filename_key]['tvec']).flatten()
        
        # Convert rotation vector to rotation matrix using Rodrigues
        R, _ = cv.Rodrigues(rvec)
        
        # Compute camera position: C = -R^T * T
        # (R^-1 = R^T for rotation matrices)
        C = -R.T @ tvec
        
        camera_positions[cam_name] = C
    
    return camera_positions

# UTILITY: Convert back to point tuples if needed
def convert_to_point_dict(voxel_dict, voxel_points_array, CAMERA_NAMES):
    """Convert index-based dict back to point-tuple-based dict matching original format"""
    point_dict = {}
    
    for idx, cam_data in voxel_dict.items():
        point = tuple(voxel_points_array[idx])
        
        # Create nested structure matching original format
        point_dict[point] = {}
        for cam in CAMERA_NAMES:
            if cam in cam_data:
                point_dict[point][cam] = cam_data[cam]
            else:
                point_dict[point][cam] = {
                    'slice_idx': [],
                    'slice_weight': []
                }
    
    return point_dict

def perform_nms(voxel_bbox_scores, voxel_bbox_associations):
    """
    Perform non-maximum suppression on voxels based on bounding box associations.
    
    For each unique combination of bounding box associations across cameras,
    only the voxel with the highest score is retained.
    
    Args:
        voxel_bbox_scores: dict mapping voxel coordinates to scores
        voxel_bbox_associations: dict mapping voxel coordinates to camera-bbox associations
        
    Returns:
        dict: filtered voxel_bbox_scores containing only maximum scoring voxels
        dict: corresponding filtered voxel_bbox_associations
    """
    # Group voxels by their bounding box association signature
    association_groups = defaultdict(list)
    
    for voxel, associations in voxel_bbox_associations.items():
        # Create a hashable key from the association pattern
        # Convert dict to tuple of (camera, bbox_id) pairs, sorted by camera
        assoc_key = tuple(sorted(
            (cam, bbox_id) for cam, bbox_id in associations.items()
        ))
        
        score = voxel_bbox_scores[voxel]
        association_groups[assoc_key].append((voxel, score))
    
    # For each group, keep only the voxel with maximum score
    filtered_scores = {}
    filtered_associations = {}
    
    for assoc_key, voxels in association_groups.items():
        # Find voxel with maximum score
        max_voxel, max_score = max(voxels, key=lambda x: x[1])
        
        filtered_scores[max_voxel] = max_score
        filtered_associations[max_voxel] = voxel_bbox_associations[max_voxel]
    
    return filtered_scores, filtered_associations

############################
# Suppression functions
############################

def perform_weighted_mean_suppression(voxel_bbox_scores, voxel_bbox_associations):
    """
    Compute the weighted mean position of voxels with the same bounding box 
    association pattern, where weights are the voxel scores.
    
    For each unique combination of bounding box associations across cameras,
    all voxels are combined into a single weighted mean position based on their scores.
    
    Args:
        voxel_bbox_scores: dict mapping voxel coordinates to scores
        voxel_bbox_associations: dict mapping voxel coordinates to camera-bbox associations
    
    Returns:
        dict: weighted mean positions (as tuples) mapping to aggregated scores
        dict: corresponding voxel_bbox_associations for each weighted mean position
    """
    # Group voxels by their bounding box association signature
    association_groups = defaultdict(list)
    for voxel, associations in voxel_bbox_associations.items():
        # Create a hashable key from the association pattern
        # Convert dict to tuple of (camera, bbox_id) pairs, sorted by camera
        assoc_key = tuple(sorted(
            (cam, bbox_id) for cam, bbox_id in associations.items()
        ))
        score = voxel_bbox_scores[voxel]
        association_groups[assoc_key].append((voxel, score))
    
    # For each group, compute weighted mean
    weighted_mean_scores = {}
    weighted_mean_associations = {}
    
    for assoc_key, voxels in association_groups.items():
        # Extract positions and weights (scores)
        positions = np.array([v[0] for v in voxels])  # Nx3 array of voxel positions
        weights = np.array([v[1] for v in voxels])    # N array of scores (used as weights)
        
        # Compute weighted mean position
        total_weight = np.sum(weights)
        weighted_mean_pos = np.sum(positions * weights[:, np.newaxis], axis=0) / total_weight
        
        # Convert back to tuple for consistency with input format
        mean_tuple = tuple(weighted_mean_pos)
        
        # Store the total weight (sum of scores) as the aggregated score
        weighted_mean_scores[mean_tuple] = total_weight
        
        # Use the association from any voxel in the group (they're all the same)
        weighted_mean_associations[mean_tuple] = voxel_bbox_associations[voxels[0][0]]
    
    return weighted_mean_scores, weighted_mean_associations