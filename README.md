# Image Super Resolution with GAN (CS236_Fall2019)
Image super resolution tries to learn a mapping between low resolution and high resolution image, so that high resolution version can be recovered. We use GAN (Generative Adversarial Network) model to achieve this.

## Files
super_res_model.py: main model file. Is used to do model training, model prediction

data_loader.py: Image loader class to load file for training, evaluation

imgutil.py: Misc image utils to do image processing

## Training data
Training data (COCO dataset) can be found at http://cocodataset.org

Download the 2017 Train images.

I just use the 512x512 images as training data.

## How to run the code
To train the model:
```
python super_res_model.py --mode train
```

To do super resolution for an image file with the model:
```
python super_res_model.py --mode train --file image_file_name
```
