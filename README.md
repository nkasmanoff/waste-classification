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


With these actions complete, all that's left is to train your model.

This is how the model I used:

```bash
$ python3 src/train.py --model-dir=models/waste-classification --epochs=250 /waste-classification/data/waste-classification
```

There are many additional arguments within this file, but I found that leaving these settings as is led to satisfactory performance
after letting this job run for roughly X hours.

In total, the best saved model (models/...) achieved a test set accuracy of Y%. We next export this model to onnx to allow for a faster and more flexible runtime.

This can be accomplished with the following code.
```bash
$ python3 src/onnx_export.py --model-dir=models/waste-classification
```



And finally we integrate this model with a real-time streaming tool, say, right in front of your kitchen garbage bin, this can be accomplished with the following command line argument made possible by building this repo out of jetson-inference.

```bash
$ imagenet.py --model=models/waste-classification/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/garbage_classification/labels.txt csi://0  --input-codec=h264 rtp://192.168.1.93:1234
```


Just open the stream from your desired RTP, and voila! Look at all that trash!
