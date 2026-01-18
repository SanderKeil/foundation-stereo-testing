import open3d as o3d
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='test_outputs_live/cloud_denoise.ply')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    print(f"Loading {args.file}...")
    pcd = o3d.io.read_point_cloud(args.file)
    
    if pcd.is_empty():
        print("Error: Point cloud is empty.")
        return

    print("Visualizing... Press 'q' or ESC to close the window.")
    print("Use the mouse to pan, rotate, and zoom.")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="FoundationStereo Result", width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Improve rendering look
    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.1, 0.1]
    opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
