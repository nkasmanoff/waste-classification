# Instantiates the jetson-inference docker container which provides all of the necessary dependencies and command line arguments.
cd ../jetson-inference
docker/run.sh --volume /home/noah/waste-classification:/waste-classification  # replace with your name
pip install wandb
