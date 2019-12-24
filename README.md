# Image Captioning
### Overview

This repository is about image captioning - generating an appropriate caption for any given image.
A deep learning model was trained for this purpose, and 
the following screenshots show the images with their generated captions. 

<img src="https://github.com/leeping-ng/ML-CV-NLP-Image-Captioning/blob/master/images/fruits.png" height='300'> 
<img src="https://github.com/leeping-ng/ML-CV-NLP-Image-Captioning/blob/master/images/nintendo.png" height='300'>

### How It Works

Image captioning mixes two deep learning domains - computer vision and natural language processing (NLP). 
Following the architecture depicted below, firstly, in the "encoder", a CNN takes in an image and produces an embedded image feature vector.
Next, in the "decoder", this image vector, as well as a word vector, is fed into a RNN, which generates the resulting caption.

<img src="https://github.com/leeping-ng/ML-CV-NLP-Image-Captioning/blob/master/images/encoder-decoder.png">

For the "encoder", a pre-trained ResNet-50 model was used for the CNN. The final fully connected layer of ResNet-50 was dropped, and replaced by an embedding layer to create the embedded image feature vector.
For the "decoder", a LSTM with one hidden layer was used for the RNN. Each generated word is fed back into the LSTM to generate the next word in the caption.
The model was trained on image-caption pairs of the Microsoft COCO dataset.
