# Instantiates the jetson-inference docker container which provides all of the necessary dependencies and command line arguments.
cd ../jetson-inference
docker/run.sh --volume /home/noah/garbage_classification:/garbage_classification  # replace with your name
