docker rm -f ros_fp
DIR=$(pwd)/../
xhost + && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name ros_fp --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $DIR:$DIR \
  -v /home:/home \
  -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  -v /media/shubhodeep/Elements/:/media/shubhodeep/Elements/ \
  --ipc=host \
  -e DISPLAY=${DISPLAY} \
  -e GIT_INDEX_FILE \
  ros_fp:latest bash -c "cd $DIR && bash"