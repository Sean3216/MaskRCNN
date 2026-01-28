# Overview
This repository is a pipeline code to train MaskRCNN and predict image with MaskRCNN. Modified input folder provides a possible variant where we use the masking produced by an anomaly detection model as extra features.

# Common Directory Structure
```
MaskRCNN/
├── backbone/
│   └── maskrcnn.py
├── data/ --> Note: The data for train or test. It's important to follow the structure. 
│   ├── <split name>/ 
│   │   │   ├── images/
│   │   │   │   └── <image files>
│   │   │   ├── labels/
│   │   │   │   └── <label files> --> Labels should be a text file that follows the expected format -> <class> <x y x y x y x y ... x y>
│   └── data.yaml --> Should follow Ultralytics formating. Refer to the current data.yaml provided
├── eval/
│   └── utils.py
├── data.py
├── losses.py
├── main.py
├── predict.py
├── train.py
└── README.md
```

# Note
Will add requirements.txt in the future