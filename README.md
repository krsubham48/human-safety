# Human Safety Using Deep Learning

## Contents

## Motivation - Why Human Detection?

## Proposed Solution

### Classification

### Detection

### Segmentation

## Convolution Neural Networks with Fully Connected Layers

### Data

### Model

## YOLO Model

### Data

### Model

## SegNet

## Results

## Conclusion

## Future Work

Future works includes use of larger dataset to bridge the gap between train accuracy and validation accuracy. Data augmentation techniques like flipping, rotation and addition of noise can be applied to further improve the results. The proposed model can be trained on new dataset containing images of large size for better results on images containing humans in various poses and posture.<br /><br />
Segmentation techniques can be further applied to get the exact mask of portion containing humans. This will add a new degree of accuracy to the proposed architecture by providing the exact boundary containing human. Instance segmentation can be further applied to distinguish between different humans present as a group in image. This will further enhance the segmentation output by providing the segmentation mask as well as the exact count of humans.
<p align="Centre">
	<img src="fw.jpg" />
</p>

## References

[1]	Jurgen Schmidhuber, *“Deep Learning in Neural Networks: An Overview,”* cIstituto Dalle Molle di Studi sull’Intelligenza Artificiale, 2014<br />
[2]	Yann LeCunn, Yoshua Bengio & Geoffrey Hinton, *“Deep Learning,”* Nature, May 2015<br />
[3]	Paul Viola and Michael Jones, *“Rapid Object Detection using a Boosted Cascade of Simple Features,”* conference on Computer Vision and Patter Recognition, 2001<br />
[4]	N. Dalal and B. Triggs, *“Histograms of oriented gradients for human detection,”* IEEE Computer Society Conference, 2005<br />
[5]	Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton, *“ImageNet Classification with Deep Convolutional Neural Networks,”* NIPS<br />
[6]	Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, *“You Only Look Once: Unified, Real-Time Object Detection,”* arXiv:1506.02640v5, May 2016<br />
[7]	Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov, *“Dropout: A Simple Way to Prevent Neural Networks from Overfitting,”* Journal of Machine Learning Research, 2014<br />
[8]	Olaf Ronneberger, Philipp Fischer, Thomas Brox, *“U-Net: Convolutional Networks for Biomedical Image Segmentation,”* accepted at MICCAI 2015<br />
[9] Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla, *"SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"*, arXiv:1511.00561v3, 10 Oct 2016