# STRLF

**Towards Spatio-Temporal Representation Learning for EEG Classification in Motor Imagery-based BCI System**

----

This is the implementation of the STRLF architecture for EEG-MI classification.

## How to use

1. Download the data from the official website.
2. Use TRLN and SRLN to extract the temporal and spatial features.
3. Input the spatial and temporal features into `fusion.m` to obtain feature importance.
4. Select features based on importance and perform classification using FCNet.

## Acknowledgment

We thank the following researchers  for their wonderful works.

Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.

Huang Z, Van Gool L. A riemannian network for spd matrix learning[C]//Proceedings of the AAAI conference on artificial intelligence. 2017, 31(1).

Wang H, Nie F, Huang H, et al. Identifying disease sensitive and quantitative trait-relevant biomarkers from multidimensional heterogeneous imaging genetics data via sparse multimodal multitask learning[J]. Bioinformatics, 2012, 28(12): i127-i136.



