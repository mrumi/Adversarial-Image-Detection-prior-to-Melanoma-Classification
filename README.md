# Adversarial-Image-Detection-prio-to-Melanoma-Classification

Data Source: International Skin Imaging Collaboration (ISIC) archive is used as data source. 

Required Python module: NumPy, SciPy, Keras, TensonFlow. GPU usage is not a mandatory. CPU can be used but it would be slow. 

The project has several stages. At first stage, three types of skin cancer was clasifies using convolutional neural network. cnn.py is the code classification using CNN. Later a set of adversarial image was generated. acgan.py contains code for this stage. Then normal and adversarial images were mixed in test dataset and the performance of CNN was tested. cnn_adverse.py - CNN was applied on adversarial images.

adverse_detection.py - code for adversarial image detection.
