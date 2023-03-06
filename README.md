# DELRFI
Deep Learning RFI mitigation code.
This repository contains the deep learning method used in the paper [paper name](paperlink) and the trained weights.

The network architecture is a U-net using the mobileNetV2 as features extractor. The data used for trains, validations and tests are pulsar observations from the Nançay radio telescope in France.

## How to use it

Download/close this repository, you can find in the table below a small description of what you can find in any file.

| filename | functions |
|----------|-----------|
|callbacks | callback to print the epoch at the end of each one |
|generate_rfi | create dynamic spectrum with rfi structures |
| metrics | modified tensorflow metric for mean IOU|
| psrlog | data generator to feed the network|
| segnet | build the network and return model |
|training | training script |
| utils | clean observations, prepare data, plots results |


#### load model

You can load the already trained model by using this line :

model = keras.models.load_model("./saved_net", custom_objects={'ConfusionMatrixMetric':MeanIOU(2)})

Disclaimer : this model has been trained with Nançay observations and may give you different results. It is recommended to train again the network with your own data. To do so, please refer to the data processing section.
