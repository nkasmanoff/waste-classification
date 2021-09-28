# waste-classification


Instantiates a classifier for real time object detection of garbage, identifying what
bin it should be in.


This work depends on the UI tools created as part of the Jetson Hello AI World Tutorial
found at https://github.com/dusty-nv/jetson-inference.


We will be using a resnet18 fine-tuned on the dataset provided, of 7 unique
categories of waste.


To reproduce this work, I am assuming you already have in your posession a Jetson
flashed with the Jetpack OS, and a camera which you can attach to that device.

Once that is all configured, ssh into your device, and replicate the following steps:


| Step | Description | Relative Path | Command |
| :---: | --- | :---: | :---: |
| 1 | Clone this repository | /home/<your-name> | git clone https://github.com/nkasmanoff/waste-classification.git  |
| 2 | Clone the jetson-inference library | /home/<your-name> | git clone --recursive https://github.com/dusty-nv/jetson-inference |
| 3 | Build the jetson-inference container, download appropriate models | /home/<your-name>/waste-classification | chmod +777 docker/setup.sh & docker/setup.sh
| 4 | Head to the newly built waste-classification repo | /waste-classification | cd /waste-classification/


With these actions complete, all that's left is


To get started here, you will need to clone this repo, along with jetson-inference, to the same repo level.


git clone X

git clone X

cd waste-classification

docker/setup.sh



And you can either download the model from HERE,

or train your own with the adapted commands

python3 train.py ...

t
python3 src/train.py --model-dir=models/waste-classification --epochs=250 /waste-classification/data/waste-classification

Next take this best model and convert it to onnx format such that it may allow for fast (and flexible) real time inference


python3 src/onnx_export.py --model-dir=models/waste-classification

To resume from a checkpoint

python3 src/train.py --model-dir=models
/waste-classification --resume=models/waste-classification/checkpoint.pth.tar --epochs=200 data/waste-classification




And to integrate this model with a real time streaming option, in the kitchen for instance, with the command (I am streaming with a CSI camera)

imagenet.py --model=models/waste-classification/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/garbage_classification/labels.txt csi://0  --input-codec=h264 rtp://192.168.1.93:1234


Just open the stream from your desired RTP, and voila! Look at all that trash!
