"""
CS236 final project: Image super resolution model
"""

from __future__ import print_function, division

import keras
from keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization, Activation, Add
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import datetime, sys, os, h5py, imageio, scipy
from data_loader import DataLoader
import numpy as np
import argparse


class Config():
    """Config class for the model training"""
    def __init__(self):
        self.channels = 3
        self.lr_h = 128
        self.lr_w = 128
        self.hr_h = 4 * self.lr_h
        self.hr_w = 4 * self.lr_w

        self.adam_lr = 0.0002
        self.adam_beta = 0.5
        self.generator_resnets = 4
        self.conv_filters = 64
        self.unet_filters = 32


class Generator():
    """Class for the Generator used in GAN"""
    def __init__(self, config):
        self.config = config

    def res_block(self, layer_input, filters):
        """residual block"""
        model = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(layer_input)
        model = Activation('relu')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Conv2D(filters, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Add()([model, layer_input])
        return model

    def deconv2d(self, layer_input, filters):
        """de conv"""
        model = UpSampling2D(size=2)(layer_input)
        model = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(model)
        model = Activation('relu')(model)
        return model

    def unet_downsample(self, layer_input, filters, f_size=4):
        """U-Net down sampling layer"""
        model = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        model = InstanceNormalization(axis=-1, center=False, scale=False)(model)
        model = Activation('relu')(model)
        return model

    def unet_upsample(self, layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """U-Net up sampling layer"""
        model = UpSampling2D(size=2)(layer_input)
        model = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(model)
        model = InstanceNormalization(axis=-1, center=False, scale=False)(model)
        model = Activation('relu')(model)
        if dropout_rate:
            model = Dropout(dropout_rate)(model)

        model = Concatenate()([model, skip_input])
        return model

    def get_generator(self):
        lr_shape = (self.config.lr_h, self.config.lr_w, self.config.channels)
        img_lr = Input(shape=lr_shape)

        # block before the res blocks
        model0 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same')(img_lr)
        model0 = Activation('relu')(model0)
        #save point
        sp0 = model0

        # res blocks
        for _ in range(self.config.generator_resnets):
            model0 = self.res_block(model0, self.config.conv_filters)

        # block after the res blocks
        model0 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(model0)
        model0 = BatchNormalization(momentum=0.8)(model0)
        model0 = Add()([model0, sp0])

        # deconv to restore to hr size
        for _ in range(2):
            model0 = self.deconv2d(model0, filters=256)

        # level 0 HR content
        gen_hr0 = Conv2D(self.config.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(model0)


        ###### Begin to apply the U-Net #######
        filters = self.config.unet_filters
        d1 = self.unet_downsample(gen_hr0, filters)
        d2 = self.unet_downsample(d1, filters * 2)
        d3 = self.unet_downsample(d2, filters * 4)
        d4 = self.unet_downsample(d3, filters * 8)

        u1 = self.unet_upsample(d4, d3, filters * 4)
        u2 = self.unet_upsample(u1, d2, filters * 2)
        u3 = self.unet_upsample(u2, d1, filters)

        u4 = UpSampling2D(size=2)(u3)

        output = Conv2D(filters=self.config.channels, kernel_size=4, strides=1,
                        padding='same', activation='tanh')(u4)

        gen_hr = output
        return Model(inputs=img_lr, outputs=gen_hr)


class Discriminator():
    """Class for the discriminator used in GAN"""
    def __init__(self, config):
        self.config = config

    def discriminator_block(self, layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        model = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        model = LeakyReLU(alpha=0.2)(model)
        if bn:
            model = BatchNormalization(momentum=0.8)(model)
        return model

    def get_discriminator(self):
        # Input img shape
        hr_shape = (self.config.hr_h, self.config.hr_w, self.config.channels)
        model = Input(shape=hr_shape)
        # save point
        sp0 = model

        filters = self.config.conv_filters
        model = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(model)
        model = LeakyReLU(alpha=0.2)(model)

        model = self.discriminator_block(model, filters, strides=2)
        model = self.discriminator_block(model, filters * 2, strides=1)

        model = self.discriminator_block(model, filters * 2, strides=2)
        model = self.discriminator_block(model, filters * 4, strides=1)

        model = self.discriminator_block(model, filters * 4, strides=2)
        model = self.discriminator_block(model, filters * 8, strides=1)
        model = self.discriminator_block(model, filters * 8, strides=2)

        model = Dense(units=filters*16)(model)
        model = LeakyReLU(alpha=0.2)(model)
        validity = Dense(1, activation='sigmoid')(model)

        return Model(sp0, validity)


class SrGanModel():
    "Super Resolution Gan Model and training"
    def __init__(self, dataset_name):
        self.config = Config()

        optimizer = keras.optimizers.Adam(self.config.adam_lr, self.config.adam_beta)

        self.dataset_name = dataset_name
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.config.hr_h, self.config.hr_w))

        # *******Discriminator******
        # in training process, will use patch gan
        self.discriminator = Discriminator(self.config).get_discriminator()
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy']) ##should try binary_crossentropy??

        # *******Generator********
        self.generator = Generator(self.config).get_generator()

        lr_shape = (self.config.lr_h, self.config.lr_w, self.config.channels)
        hr_shape = (self.config.hr_h, self.config.hr_w, self.config.channels)
        img_lr = Input(shape=lr_shape)
        img_hr = Input(shape=hr_shape)

        fake_hr = self.generator(img_lr)

        # discriminator not trainable when training generator
        self.discriminator.trainable = False
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_hr])
        self.combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=optimizer)


    def train(self, iterations, batch_size=1, gen_interval=100):

        for iteration in range(iterations):

            # patch gan
            patch_size = int(self.config.hr_h / 16)
            patch = (patch_size, patch_size, 1)

            #  **********Train Discriminator**********
            imgs_hr, imgs_lr, imgs_name = self.data_loader.load_data(batch_size)
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + patch)
            fake = np.zeros((batch_size,) + patch)

            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            print("d_loss_real={}, d_loss_fake={}".format(d_loss_real, d_loss_fake))

            #  ************Train Generator***********
            imgs_hr, imgs_lr, imgs_name = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + patch)
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, imgs_hr])
            print("g_loss={}".format(g_loss))

            if iteration % gen_interval == 0:
                self.sample_images(iteration)

            if iteration == 1000 or iteration == 4000 or iteration == 8000 or iteration == 14000:
                self.save_model(iteration)

    def sample_images(self, iteration):
        os.makedirs('images/{}'.format(self.dataset_name), exist_ok=True)

        imgs_hr, imgs_lr, imgs_name = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        fake_hr = 0.5 * fake_hr + 0.5
        imageio.imwrite("images/{}/{}_{}_fakehr0.png".format(self.dataset_name, iteration, imgs_name[0]), fake_hr[0])
        imageio.imwrite("images/{}/{}_{}_fakehr1.png".format(self.dataset_name, iteration, imgs_name[1]), fake_hr[1])

    def save_model(self, model_num):
        self.generator.save("saved_model/model_generator_{}.h5".format(model_num))
        self.discriminator.save("saved_model/model_discriminator_{}.h5".format(model_num))
        self.combined.save("saved_model/model_combined_{}.h5".format(model_num))


def load_model(model_name):
    """Load model from saved file"""
    model = keras.models.load_model("saved_model/{}.h5".format(model_name),
                                    custom_objects={'InstanceNormalization': InstanceNormalization})
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN Trainig')
    parser.add_argument('--mode', default='', help='running mode')
    parser.add_argument('--file', default='', help='image file to process')

    args = parser.parse_args()

    if args.mode == 'train':
        # change the dataset name here if working on different dataset
        gan = SrGanModel(dataset_name='coco512')
        gan.train(iterations=15000, batch_size=1, gen_interval=100)
    elif args.mode == 'predict':
        generator = load_model("model_generator_8000")
        generator.summary()

        # change the dataset name here if working on different dataset
        data_loader = DataLoader(dataset_name='set14', img_res=(512, 512))
        img_file = args.file
        if not img_file:
            imgs_hr, imgs_lr, imgs_name = data_loader.load_data(2)
            fake_hr = generator.predict(imgs_lr)

            imageio.imwrite("./{}_fakehr0.png".format(imgs_name[0]), fake_hr[0])
            imageio.imwrite("./{}_fakehr1.png".format(imgs_name[1]), fake_hr[1])
        else:
            imgs_hr, imgs_lr, imgs_name = data_loader.load_image(img_file, is_testing=True)
            imgs_lr = [imgs_lr]
            imgs_lr = np.array(imgs_lr) / 127.5 - 1.
            fake_hr = generator.predict(imgs_lr)

            imageio.imwrite("./{}_fakehr0.png".format(imgs_name), fake_hr[0])
    else:
        print('Please provide running mode "--mode train/predict --file"')
