# cGan-Based-Manga-Colorization-using-1-training-image

**Currently still a work in progress**

Currently getting varying results when using just a single training image. Most manga characters arent as clean as the ones demonstrated in the paper. Works best if target Image has little to no screentones.

Paper: https://arxiv.org/pdf/1706.06918.pdf

TODO:

- [ ] ScreenTone Removal
- [ ] Reapplying Screentones/Shading
- [ ] Fix Trapped Ball Segmentation
- [ ] Test on simpler Datasets

# Current Results on Goku

### Input Image

![alt text](bw_image.png "Input Image")

### cGAN generated Image

![alt text](gen_image.png "Gan Image")

### Post Processed Image

![alt text](processed_image.png "Post Processed Image")

### Original Image

![alt text](test_images/2.jpg "Original Image")
