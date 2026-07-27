set -euo pipefail

DOCKER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR=$(cd "$DOCKER_DIR/.." && pwd)
IMAGE_NAME="foundationpose:grpc"

if [ -z "${SSH_AUTH_SOCK:-}" ]; then
  echo "SSH_AUTH_SOCK is not set - start an ssh-agent and ssh-add a key with access to FuturHandRobotics/futur_grpc first." >&2
  exit 1
fi

DOCKER_BUILDKIT=1 docker build --network host --ssh default -f "$DOCKER_DIR/dockerfile.grpc" -t "$IMAGE_NAME" "$DOCKER_DIR"

docker rm -f foundationpose
xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE $IMAGE_NAME bash -c "cd $DIR && bash"
