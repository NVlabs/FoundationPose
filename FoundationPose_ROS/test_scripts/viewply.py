import open3d as o3d

# Load the .ply file
ply_path = "/home/shubhodeep/Figueroa_Lab/Foundation_Pose/FoundationPose_ROS/outputs/scene_complete.ply"
point_cloud = o3d.io.read_point_cloud(ply_path)

# Check if the point cloud is loaded properly
if point_cloud.is_empty():
    print("Failed to load the point cloud.")
else:
    # Create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    # Visualize the point cloud along with the coordinate frame
    o3d.visualization.draw_geometries([point_cloud],
                                      window_name="PLY Point Cloud Viewer with Axes",
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50,
                                      point_show_normal=False,
                                      mesh_show_wireframe=False,
                                      mesh_show_back_face=False)

# If you want to save the visualized window as an image, use this:
def save_view_image(vis):
    vis.capture_screen_image("point_cloud_view_with_axes.png")
    return False

# Uncomment below lines to enable saving a screenshot of the visualization
# vis = o3d.visualization.VisualizerWithKeyCallback()
# vis.create_window()
# vis.add_geometry(point_cloud)
# vis.add_geometry(coordinate_frame)
# vis.register_key_callback(ord("S"), save_view_image)
# vis.run()
# vis.destroy_window()
