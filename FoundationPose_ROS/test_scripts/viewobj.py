import open3d as o3d

# Load the .obj file
obj_path = "/home/shubhodeep/Figueroa_Lab/Foundation_Pose/FoundationPose/demo_data/new_data/mesh/textured_mesh.obj"
mesh = o3d.io.read_triangle_mesh(obj_path)

# Check if the mesh is loaded properly
if not mesh.has_triangles():
    print("Failed to load the mesh.")
else:
    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh, coordinate_frame],
                                      window_name="OBJ Viewer",
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50,
                                      mesh_show_wireframe=True)

# If you want to save the visualized window as an image, use this:
def save_view_image(vis):
    vis.capture_screen_image("mesh_view.png")
    return False
