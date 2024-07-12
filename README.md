# Hierarchical-Color-constancy-via-Efficient-Spectral-Feature-Extraction

<img src="https://github.com/dongkeun3520/Hierarchical-Color-constancy-via-Efficient-Spectral-Feature-Extraction/blob/main/Images/proposed_network.png" />

# Summary

Our model is designed to estimate a single illuminant of an RGB image. Our methodleverages a pretrained model to estimate a multi-spectral image from an RGB image. Our model then extracts spectral features from the generated multi-spectral images.  Despite the generated multi-spectral image being artificial, the results indicate its effectiveness.

# HSI2MS

In addition to the ICVL dataset, many hyperspectral datasets consist of a substantial number of bands. While a greater number of bands provides more information in the dataset, generating hyperspectral images from RGB is a challenging task. To balance the considerations of the number of bands and the trade-off in generation accuracy, the conversion of hyperspectral images to 8 channels has been undertaken. The file 'HSI2MS.py' is responsible for converting hyperspectral images into 8-channel images

# RGB2MS

RGB2MS model is trained for generating multi-spectral image from RGB image. This model is trained using the ICVL dataset, taking into account the subject or landscape of the color constancy dataset. Pretrained weight can be downloaded below.


# Pretrained weights

RGB2MS weight : [Download pretrained model weight](https://drive.google.com/file/d/1MSpuWtSVk9KqdSQYW94DTVChgKKD9MY4/view?usp=drive_link)

Model weight trained with NUS-8 dataset :  [Download pretrained model weight](https://drive.google.com/file/d/1ymbYKhLFFqNELdMZNObbIxNid-Fdw3en/view?usp=drive_link)
