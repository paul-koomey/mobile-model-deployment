# mobile-model-deployment

This model that this project takes in a batch of 64x64 pixel H&E slide images, uses an encoder to compress them into a latent spaces, and then uses a decoder to regenerate the original images from their respective latent spaces. This specific implementation of the VAE will utilize only the encoder portion of the model to compress an H&E slide image to a latent space on mobile so that it can be sent over a mobile/WiFi network and 'uncompressed' using the decoder portion of the model on a different device.


## model_to_ONNX_notebook directory

This directory contains the code required to convert a PyTorch Lightning model into an ONNX model that can be deployed to mobile. After being converted to ONNX, the model will then be compressed using quantization.


## android_app directory

This directory contains the android studio code for the mobile application that will be 'compressing' images and sending latent spaces.

