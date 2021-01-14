# STL10
Image classification using PyTorch on (almost) STL-10 dataset

Loading data from a locally stored folder "dataset" (added to .gitignore).
Dataset is almost similar to STL-10:
https://ai.stanford.edu/~acoates/stl10/

The difference is that images are resampled for each of 10 classes:
* local dataset has 1000 train and 300 test images per class;
* original STL-10 has 500 train and 800 test images per class.
