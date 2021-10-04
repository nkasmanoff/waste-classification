# Instantiates the jetson-inference docker container which provides all of the necessary dependencies and command line arguments.
cd ../jetson-inference
docker/run.sh --v /home/noah/waste-classification:/waste-classification  -c "cd /waste-classification && pip install wandb"
