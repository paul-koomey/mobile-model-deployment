# mobile-model-deployment

Currently a work in progress, this project will convert the encoder portion of an exisiting variational autoencoder (VAE) model (from this [preprint](https://arxiv.org/abs/2205.10373)) to ONNX so that it can be deployed on a mobile device. The VAE will compress an H&E slide image to a latent space  so that it can be sent over a mobile/WiFi network and 'uncompressed' using the decoder portion of the model on a different device.


## model_to_ONNX_notebook directory

This directory contains the code required to convert a PyTorch Lightning model into an ONNX model that can be deployed to mobile.
After being converted to ONNX, it will be quantized.


## android_app directory

This directory contains the android studio code for the mobile application.

