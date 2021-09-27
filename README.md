# waste-classification


Instantiates my Jetson Nano project, a camera classification tool


This work depends on the UI tools created as part of the Jetson Hello AI World Tutorial
found at https://github.com/dusty-nv/jetson-inference.


To get started here, you will need to clone this repo, along with jetson-inference, to the same repo level.


git clone X

git clone X

cd waste-classification

docker/setup.sh



And you can either download the model from HERE,

or train your own with the adapted commands

python3 train.py ...


python3 src/train.py --model-dir=models/waste-classification data/garbage_classification

Next take this best model and convert it to onnx format such that it may allow for fast (and flexible) real time inference


python3 src/onnx_export.py --model-dir=models/waste-classification


And to integrate this model with a real time streaming option, in the kitchen for instance, with the command (I am streaming with a CSI camera)

imagenet.py --model=models/waste-classification/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/garbage_classification/labels.txt csi://0  --input-codec=h264 rtp://192.168.1.93:1234


Just open the stream from your desired RTP, and voila! Look at all that trash!
