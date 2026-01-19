import open3d as o3d
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--output', type=str, default='snapshot.png')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    pcd = o3d.io.read_point_cloud(args.file)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720) # Off-screen
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.1, 0.1]
    opt.point_size = 2.0
    
    # Set view control for a better angle provided we know the camera setup
    ctr = vis.get_view_control()
    # Basic front view
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)
    
    # Render and save
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(args.output)
    vis.destroy_window()
    print(f"Saved snapshot to {args.output}")

if __name__ == "__main__":
    main()
