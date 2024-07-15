import argparse
from point_cloud_processing import process_point_cloud
from cube_generator import generate_cube

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Processing and Cube Generation Tool")
    parser.add_argument("--process", action="store_true", help="Process point cloud")
    parser.add_argument("--generate-cube", action="store_true", help="Generate cube point cloud")
    parser.add_argument("--input", type=str, help="Input point cloud file")
    parser.add_argument("--output", type=str, help="Output file name")
    parser.add_argument("--cube-size", type=int, help="Size of the cube to generate")
    
    # New arguments for point cloud processing
    parser.add_argument("--initial-voxel-size", type=float, default=0.1, help="Initial voxel size for downsampling")
    parser.add_argument("--normal-radius", type=float, default=5, help="Radius for normal estimation")
    parser.add_argument("--normal-max-nn", type=int, default=30, help="Max neighbors for normal estimation")
    parser.add_argument("--downsample-batch-size", type=int, default=200000, help="Batch size for GPU downsampling")
    parser.add_argument("--inside-downsample-voxel-size", type=float, default=1, help="Voxel size for inside point downsampling")
    parser.add_argument("--edge-threshold-percentile", type=float, default=0.9, help="Percentile for edge detection")
    parser.add_argument("--ball-pivoting-radii", type=float, nargs='+', default=[0.1, 1], help="Radii for ball pivoting algorithm")
    parser.add_argument("--edge-output", type=str, help="Output file for edge points")
    parser.add_argument("--inside-output", type=str, help="Output file for inside points")
    parser.add_argument("--combined-output", type=str, help="Output file for combined points")

    args = parser.parse_args()

    if args.process:
        if not args.input or not args.output:
            print("Error: Input and output files are required for processing.")
            return
        process_point_cloud(
            args.input,
            args.output,
            initial_voxel_size=args.initial_voxel_size,
            normal_radius=args.normal_radius,
            normal_max_nn=args.normal_max_nn,
            downsample_batch_size=args.downsample_batch_size,
            inside_downsample_voxel_size=args.inside_downsample_voxel_size,
            edge_threshold_percentile=args.edge_threshold_percentile,
            ball_pivoting_radii=args.ball_pivoting_radii,
            edge_filename=args.edge_output,
            inside_filename=args.inside_output,
            combined_filename=args.combined_output
        )
    elif args.generate_cube:
        if not args.cube_size or not args.output:
            print("Error: Cube size and output file are required for cube generation.")
            return
        generate_cube(args.cube_size, args.output)
    else:
        print("Error: Please specify either --process or --generate-cube")

if __name__ == "__main__":
    main()
