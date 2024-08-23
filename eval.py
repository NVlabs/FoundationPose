
from datareader import OrganaReader
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
mesh_files = ['beaker_250ml.obj', 'conical_flask_500ml.obj','conical_flask_250ml.obj', 'beaker_30ml.obj']
meshes = [f'{code_dir}/perception_data/objects/{mesh}' for mesh in mesh_files]
parser.add_argument('--mesh_files', type=str, nargs='+', default=meshes)
parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/perception_data/black_paper/1')
args = parser.parse_args()
reader = OrganaReader(base_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
est_pose_beaker_30ml_files = []
est_pose_beaker_250ml_files = []
est_pose_conical_flask_250ml_files = []
est_pose_conical_flask_500ml_files = []
for est_pose in reader.est_pose_files:
    if 'beaker_30ml' in est_pose:
        est_pose_beaker_30ml_files.append(est_pose)
    elif 'beaker_250ml' in est_pose:
        est_pose_beaker_250ml_files.append(est_pose)
    elif 'conical_flask_250ml' in est_pose:
        est_pose_conical_flask_250ml_files.append(est_pose)
    elif 'conical_flask_500ml' in est_pose:
        est_pose_conical_flask_500ml_files.append(est_pose)
print(est_pose_beaker_30ml_files)

def parse_ground_truth_pose(file_path):
    """
    Parse the ground truth pose from a text file in the specified format.
    
    :param file_path: Path to the text file containing the ground truth pose.
    :return: A 1x3 numpy array of the translation vector.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Assuming the translation vector is in the third line (first three values)
    translation = np.array([float(val) for val in lines[2].strip().split()])
    
    return translation

def extract_estimated_position(matrix):
    """
    Extract the estimated position from the first row of a 4x3 matrix.
    
    :param matrix: A 4x4 numpy array.
    :return: A 1x3 numpy array of the estimated position.
    """
    # retrieve the first three elements of the last column
    return matrix[:3, 3]

def calculate_mae(gt_positions, est_positions):
    """
    Calculate the Mean Absolute Error (MAE) between ground truth and estimated positions.
    
    :param gt_positions: List of ground truth positions (numpy arrays).
    :param est_positions: List of estimated positions (numpy arrays).
    :return: Mean Absolute Error (MAE) as a float.
    """
    print(gt_positions)
    print(est_positions)
    # Compute absolute errors for positions
    abs_errors = [np.linalg.norm(gt - est) for gt, est in zip(gt_positions, est_positions)]
    
    # Compute mean across all position errors
    mae = np.mean(abs_errors)
    
    return mae

def evaluate():
    gt_files = {
        'beaker_30ml': '2.tf',
        'beaker_250ml': '5.tf',
        'conical_flask_250ml': '14.tf',
        'conical_flask_500ml': '17.tf',
    }

    gt_positions = {}
    for obj, file_name in gt_files.items():
        file_path = os.path.join(args.test_scene_dir, file_name)
        gt_positions[obj] = parse_ground_truth_pose(file_path)

    # Assuming reader.est_pose_{object}_files are lists of file paths to the 4x3 matrices
    est_poses = {
        'beaker_250ml': [np.load(file) for file in est_pose_beaker_250ml_files],
        'conical_flask_500ml': [np.load(file) for file in est_pose_conical_flask_500ml_files],
        'conical_flask_250ml': [np.load(file) for file in est_pose_conical_flask_250ml_files],
        'beaker_30ml': [np.load(file) for file in est_pose_beaker_30ml_files],
    }

    # Calculate and print MAE for each object
    for obj in gt_files.keys():
        est_positions = [extract_estimated_position(pose) for pose in est_poses[obj]]
        mae = calculate_mae([gt_positions[obj]], est_positions)
        print(f'MAE for {obj}: {mae}')

# Parse arguments
args = parser.parse_args()

# Call evaluate to perform the calculation
evaluate()
