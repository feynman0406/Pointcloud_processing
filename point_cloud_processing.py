import open3d as o3d
import numpy as np
import time
# Read las file to gpu tensor

def las_to_pcd_gpu(las_file, gpu=False):
    """
    Convert a LAS file to an Open3D PointCloud, optionally transferring it to the GPU.

    Parameters:
    las_file (str): Path to the input LAS file.
    gpu (bool): If True, transfer the point cloud to the GPU.

    Returns:
    o3d.geometry.PointCloud or o3d.t.geometry.PointCloud: The resulting point cloud.
    """
    start_time = time.time()

    try:
        # Read LAS file
        las = laspy.read(las_file)
        e1 = time.time()
        print(f'LAS file read in {e1 - start_time:.6f} seconds')

        # Extract point coordinates
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # Check if color information exists
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            colors = np.vstack((las.red, las.green, las.blue)).transpose().astype(np.float64)
            # Normalize colors if they are in the range [0, 65535]
            if np.max(colors) > 1.0:
                colors /= 65535.0
        else:
            colors = None

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Convert to Open3D Tensor and specify the device
        if gpu:
            device = o3c.Device("CUDA:0")
            pcd_gpu = o3d.t.geometry.PointCloud.from_legacy(pcd, device=device)
            output = pcd_gpu
        else:
            output = pcd

        return output
    
    except Exception as e:
        print(f"Error processing LAS file: {e}")
        return None



#Map color to mesh
def map_colors_to_mesh(pcd, mesh, device="cuda:0", k=1):
    """
    Map colors from a point cloud to a mesh using GPU-based k-NN search.
    
    Parameters:
    pcd (o3d.geometry.PointCloud): The legacy point cloud with color information.
    mesh (o3d.geometry.TriangleMesh): The mesh to which colors will be mapped.
    device (str): The device to use for computation ("cpu:0" or "cuda:0").
    k (int): The number of nearest neighbors to consider.
    
    Returns:
    o3d.geometry.TriangleMesh: The mesh with mapped colors.
    """
    try:
        # Ensure inputs are correct types
        if not isinstance(pcd, o3d.geometry.PointCloud) or not isinstance(mesh, o3d.geometry.TriangleMesh):
            raise TypeError("Input pcd must be legacy PointCloud and mesh must be legacy TriangleMesh")

        # Check if point cloud has colors
        if not pcd.has_colors():
            raise ValueError("Point cloud does not have colors")

        # Convert point cloud and mesh to tensor-based geometries
        pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd)
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        # Move data to specified device
        device = o3c.Device(device)
        pcd_points = pcd_t.point["positions"].to(device)
        pcd_colors = pcd_t.point["colors"].to(device)
        mesh_vertices = mesh_t.vertex["positions"].to(device)

        print(f"PCD points shape: {pcd_points.shape}")
        print(f"PCD colors shape: {pcd_colors.shape}")
        print(f"Mesh vertices shape: {mesh_vertices.shape}")

        # Perform GPU-based k-NN search
        indices, _ = knn_search_gpu(pcd_points, mesh_vertices, k)

        # Map colors from nearest neighbors
        start_time = time.time()
        nearest_colors = pcd_colors[indices]
        if k > 1:
            nearest_colors = nearest_colors.mean(dim=1)
        
        # Ensure the color tensor has the correct shape
        if nearest_colors.shape != mesh_vertices.shape:
            print(f"Reshaping colors from {nearest_colors.shape} to {mesh_vertices.shape}")
            nearest_colors = nearest_colors.reshape(mesh_vertices.shape)

        mesh_t.vertex["colors"] = nearest_colors
        print(f"Color mapping time: {time.time() - start_time:.4f} seconds")

        # Convert back to legacy mesh
        colored_mesh = mesh_t.to_legacy()

        return colored_mesh

    except Exception as e:
        print(f"Error in map_colors_to_mesh: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



# GPU-based k-NN search function
def knn_search_gpu(dataset_points, query_points, k):
    print(f"Running on GPU")

    # Create NearestNeighborSearch object and build index
    nns = o3c.nns.NearestNeighborSearch(dataset_points)
    nns.knn_index()

    # Perform k-NN search and measure time
    start_time = time.time()
    indices, distances = nns.knn_search(query_points, k)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"GPU execution time: {execution_time:.6f} seconds")

    return indices, distances
# Example usage
#points_tensor = o3d.t.geometry.PointCloud(o3c.Tensor(points, device=o3c.Device("CUDA:0"))    
#indices, distances = knn_search_gpu(points_tensor, vertices_tensor, k)    






#save_points_to_las
def save_points_to_las(extracted_points, filename="extracted_points_near_edges.las"):
    try:
        # Calculate offsets and scales
        min_vals = np.min(extracted_points, axis=0)
        max_vals = np.max(extracted_points, axis=0)
        ranges = max_vals - min_vals
        scales = ranges / 10000  # Adjust this value based on precision needs
        
        # Create LAS header
        las_header_extracted = laspy.LasHeader(point_format=3, version="1.2")
        las_header_extracted.offsets = min_vals
        las_header_extracted.scales = scales

        # Create LAS data object
        las_extracted = laspy.LasData(las_header_extracted)
        las_extracted.x = extracted_points[:, 0]
        las_extracted.y = extracted_points[:, 1]
        las_extracted.z = extracted_points[:, 2]

        # Write LAS file
        las_extracted.write(filename)
        print(f"Successfully wrote {filename}")

    except Exception as e:
        print(f"Error writing LAS file: {e}")
#save_points_to_las(points)



# GPU voxelization
# Function to perform batched voxel downsampling on the GPU
def voxel_down_sample_gpu_batched(pcd, voxel_size, batch_size):
    num_points = pcd.point.positions.shape[0]
    num_batches = (num_points + batch_size - 1) // batch_size

    downsampled_points_list = []
    downsampled_colors_list = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_points)
        
        # Timing the creation of the batch point cloud
        start_batch = time.time()
        batch_pcd = o3d.t.geometry.PointCloud(pcd.point.positions[start_idx:end_idx])
        if pcd.point.colors is not None:
            batch_pcd.point.colors = pcd.point.colors[start_idx:end_idx]
        #end_batch = time.time()
        #print(f"Batch {i+1} creation time: {end_batch - start_batch:.6f} seconds")

        # Timing the voxel downsampling of the batch
        start_downsample = time.time()
        batch_downsampled = batch_pcd.voxel_down_sample(voxel_size=voxel_size)
        #end_downsample = time.time()
        #print(f"Batch {i+1} voxel downsampling time: {end_downsample - start_downsample:.6f} seconds")
        
        downsampled_points_list.append(batch_downsampled.point.positions.to(o3c.Device("CPU:0")))
        if batch_downsampled.point.colors is not None:
            downsampled_colors_list.append(batch_downsampled.point.colors.to(o3c.Device("CPU:0")))

    # Timing the conversion of Open3D tensors to CuPy arrays
    start_conversion_to_cupy = time.time()
    cupy_points_list = [cp.asarray(p.numpy()) for p in downsampled_points_list]
    end_conversion_to_cupy = time.time()
    print(f"Conversion to CuPy arrays time: {end_conversion_to_cupy - start_conversion_to_cupy:.6f} seconds")
    
    # Concatenate the points
    start_concat_points = time.time()
    concatenated_points_cupy = cp.concatenate(cupy_points_list, axis=0)
    end_concat_points = time.time()
    print(f"Concatenation time for points: {end_concat_points - start_concat_points:.6f} seconds")

    # Timing the conversion back to Open3D tensor for points
    start_conversion_to_open3d_points = time.time()
    concatenated_points_tensor = o3c.Tensor(cp.asnumpy(concatenated_points_cupy), device=o3c.Device("CUDA:0"))
    end_conversion_to_open3d_points = time.time()
    print(f"Conversion back to Open3D tensor for points time: {end_conversion_to_open3d_points - start_conversion_to_open3d_points:.6f} seconds")
    
    if downsampled_colors_list:
        start_conversion_to_cupy_colors = time.time()
        cupy_colors_list = [cp.asarray(c.numpy()) for c in downsampled_colors_list]
        end_conversion_to_cupy_colors = time.time()
        print(f"Conversion to CuPy arrays time for colors: {end_conversion_to_cupy_colors - start_conversion_to_cupy_colors:.6f} seconds")

        # Concatenate the colors
        start_concat_colors = time.time()
        concatenated_colors_cupy = cp.concatenate(cupy_colors_list, axis=0)
        end_concat_colors = time.time()
        print(f"Concatenation time for colors: {end_concat_colors - start_concat_colors:.6f} seconds")

        # Timing the conversion back to Open3D tensor for colors
        start_conversion_to_open3d_colors = time.time()
        concatenated_colors_tensor = o3c.Tensor(cp.asnumpy(concatenated_colors_cupy), device=o3c.Device("CUDA:0"))
        end_conversion_to_open3d_colors = time.time()
        print(f"Conversion back to Open3D tensor for colors time: {end_conversion_to_open3d_colors - start_conversion_to_open3d_colors:.6f} seconds")

        concatenated_pcd = o3d.t.geometry.PointCloud(concatenated_points_tensor)
        concatenated_pcd.point.colors = concatenated_colors_tensor
    else:
        concatenated_pcd = o3d.t.geometry.PointCloud(concatenated_points_tensor)
    
    return concatenated_pcd


#Example of uasage
#pcd_gpu_downsampled = voxel_down_sample_gpu_batched(pcd_gpu, voxel_size=0.01, batch_size=400000)
