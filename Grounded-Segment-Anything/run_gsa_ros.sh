docker rm -f gsa_ros

docker run --gpus all -it --rm --net=host --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v "${PWD}":/home/appuser/Grounded-Segment-Anything \
	-e DISPLAY=$DISPLAY \
	--name=gsa_ros \
	--ipc=host -it ghcr.io/shubho-upenn/gsa_ros:latest
