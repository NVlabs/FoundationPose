import os
code_dir = os.path.dirname(os.path.realpath(__file__))
mesh_files = ['cube.obj','beaker_250ml.obj']
meshes = [f'{code_dir}/perception_data/objects/{mesh}' for mesh in mesh_files]
