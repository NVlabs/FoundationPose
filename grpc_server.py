#!/usr/bin/env python3
from estimater import *
import argparse
from concurrent import futures

import grpc
from futur_grpc import (
    Pose,
    Empty,
    PoseEstimateServicer,
    add_PoseEstimateServicer_to_server,
)


class PoseEstimateService(PoseEstimateServicer):
  def __init__(self, est, est_refine_iter, track_refine_iter):
    self.est = est
    self.est_refine_iter = est_refine_iter
    self.track_refine_iter = track_refine_iter

  def _unpack_frame(self, request):
    rgb = np.frombuffer(request.rgb_data, dtype=np.uint8).reshape(request.height, request.width, 3)
    depth = np.frombuffer(request.depth_data, dtype=np.float32).reshape(request.height, request.width)
    K = np.array(request.intrinsics, dtype=np.float64).reshape(3, 3)
    return rgb, depth, K

  def TrackStream(self, request_iterator, context):
    for request in request_iterator:
      rgb, depth, K = self._unpack_frame(request)

      if self.est.pose_last is None:
        if not request.HasField('mask'):
          logging.warning('no pose to track from and no mask provided to register; skipping frame')
          yield Pose(success=False)
          continue
        mask = np.frombuffer(request.mask, dtype=np.uint8).reshape(request.height, request.width).astype(bool)
        pose = self.est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)
      else:
        pose = self.est.track_one(rgb=rgb, depth=depth, K=K, iteration=self.track_refine_iter)

      yield Pose(
          translation=pose[:3, 3].tolist(),
          rotation=pose[:3, :3].reshape(-1).tolist(),
          success=True,
      )
    #Reset the stream to prevent stale state from hanging around
    self.est.pose_last = None

  def Reset(self, request, context):
    self.est.pose_last = None
    return Empty()

  def SeedPose(self, request, context):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(request.rotation, dtype=np.float32).reshape(3, 3)
    pose[:3, 3] = np.array(request.translation, dtype=np.float32)
    self.est.pose_last = torch.as_tensor(pose, device='cuda', dtype=torch.float)
    return Empty()


if __name__ == '__main__':
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('mesh_file', type=str, nargs='?', help='path to the object mesh to load (e.g. textured_simple.obj)',
                      default=f'{code_dir}/meshes/Perfectionist_Pro_Puck.obj')
  parser.add_argument('--port', type=int, default=50051)
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{os.path.dirname(os.path.realpath(__file__))}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  if mesh.visual.kind == 'texture' and mesh.visual.material.image is None:
      try:
          color = mesh.visual.material.diffuse[:3]
      except Exception:
          color = [128, 128, 128]
      tex = Image.new('RGB', (16, 16), tuple(int(c) for c in color))
      mesh.visual.material.image = tex

  if mesh.visual.uv is None:
      # dummy UVs — since the texture is a flat solid color, it doesn't matter
      # what they point to, they just need to exist and be the right shape
      mesh.visual.uv = np.zeros((len(mesh.vertices), 2), dtype=np.float32)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
      model_pts=mesh.vertices,
      model_normals=mesh.vertex_normals,
      mesh=mesh,
      scorer=scorer,
      refiner=refiner,
      debug_dir=args.debug_dir,
      debug=args.debug,
      glctx=glctx,
  )
  logging.info('estimator initialization done')

  server = grpc.server(
      futures.ThreadPoolExecutor(max_workers=10),
      options=[
          ('grpc.max_send_message_length', 64 * 1024 * 1024),
          ('grpc.max_receive_message_length', 64 * 1024 * 1024),
      ],
  )
  add_PoseEstimateServicer_to_server(
      PoseEstimateService(est, args.est_refine_iter, args.track_refine_iter), server
  )
  server.add_insecure_port(f'[::]:{args.port}')
  server.start()
  logging.info(f'listening on {args.port}')
  server.wait_for_termination()
