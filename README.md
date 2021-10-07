# waste-classification


Instantiates a classifier for real time object detection of waste, identifying what
bin it should be in.

This work depends on the UI tools created as part of the Jetson Hello AI World Tutorial
found at https://github.com/dusty-nv/jetson-inference. This work assumes familiarity with at least the first 3 episodes
of that tutorial, as we will be taking advantage of that format for training to inference procedure.

Additionally this work is a bit of an introduction to various MLOps frameworks (inspired by https://github.com/graviraja/MLOps-Basics) and as such introduces to the code Weights and Biases logging, and Hydra config files for improved hyper-parameter tuning.

From a technical perspective, this work fine-tunes a Resnet18 (initially trained on ImageNet), to categorize 7 unique
classes of waste.


The data we use for this task is adapted from two sources, https://www.kaggle.com/techsash/waste-classification-data and https://www.kaggle.com/asdasdasasdas/garbage-classification.

These data streams give us roughly a training-validation-testing split of 300-80-80 images per class, which are:

        - cardboard
        - glass
        - metal
        - organic
        - paper
        - plastic
        - trash

Understandably, having 7 different waste bins might be a little excessive, so it is certainly feasible to pool together some of these categories.
I will leave that to a future work, since having more classes at least to start is not a bad thing!



To reproduce this work, I am assuming you already have in your possession a Jetson
flashed with the Jetpack OS, and a camera which you can attach to that device.

Once that is all configured, ssh into your device, and replicate the following steps:


| Step | Description | Relative Path | Command |
| :---: | --- | :---: | :---: |
| 1 | Clone this repository | /home/<your-name> | git clone https://github.com/nkasmanoff/waste-classification.git  |
| 2 | Clone the jetson-inference library | /home/<your-name> | git clone --recursive https://github.com/dusty-nv/jetson-inference |
| 3 | Build the jetson-inference container, download appropriate models + packages | /home/your-name/waste-classification | chmod +777 docker/setup.sh & docker/
| 4 | Head back to this repo, download appropriate models + packages | /home/jetson-inference | cd /waste-classification & pip install -r requirements.txt


I would be eager to include step 4 as part of setup.sh, but currently unaware how to do so. Please submit a PR if you know how to do so!

With these actions complete, all that's left is to train your model.

This an example command which leaves all hyper-parameters fixed, except instantiates two training runs of different learning rates to train for 100 epochs.

```bash
$ python3 src/train.py --m lr=.001,0001 epochs=30
```

There are many additional arguments within '''train.py''', but I found that leaving these settings as is led to test set top1 accuracy of ~63% and top5 accuracy of ~98%  after letting this job run for roughly 24 hours. I tried several ways to speed this performance up, but none seemed to help. If anyone has ideas as to how we can remove some of these plateaus in the curve, I would be happy to chat :-).

We next export this model to onnx to allow for a faster and more flexible runtime.

This can be accomplished with the following code.
```bash
$ python3 src/onnx_export.py --model-dir=models/waste-classification
```



And finally we integrate this model with a real-time streaming tool, say, right in front of your kitchen garbage bin, this can be accomplished with the following command line argument made possible by building this repo out of jetson-inference.

```bash
$ imagenet.py --model=models/waste-classification/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/waste-classification/labels.txt csi://0  --input-codec=h264 rtp://<YOUR IP ADDRESS>:1234
```


Just open the stream from your desired RTP, and voila! Look at all that trash!


# Next Steps

For more detail on this project, please check out https://nkasmanoff.github.io/blog.html, where at the end I point out a few ideas to extend
the use-cases of a model like this.
