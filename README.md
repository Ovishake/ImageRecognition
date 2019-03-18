# ImageRecognition
Using DataFlow pipelines from pandas to do classification over AlexNet like CNN. 
Using TENSORFLOW-KERAS and SCI-KIT LEARN on NVIDIA K80 GPU by the University at Albany. 
Over the NIH CHEST X-RAY Data.set for classification of Diseases.

Primary File:
NIH.py

API used:
Tensorflow
Keras
Pandas
Scikit-Learn

About the data:
The data was obtained from NIH wesbsite. And when Unzipped they all fell into the
Images folder in .png format. Some of the data had 3 dimension. Unlike, my course
work where I did cv2.imread, I have used Keras to make a continouos flow of Images
using the flow_from_dataframe.

Other files:
nihtf.py is a pure tensorflow implementation using the DEPRECATED tf layers API
as they are deprecated in TF2.0. The default layer building API for Tensorflow
from Tensorflow 2.0 is TENSORFLOW-KERAS.
