import tensorflow as tf
import imageio
import skimage
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import glob, os, argparse


def produce_training_data():
    image_files = glob.glob("/Users/wkang/Downloads/SR_testing_datasets/Set14/*.png")

    count = 0
    size = 512  # 400
    for image_file in image_files:
        print(image_file)
        img = imageio.imread(image_file, pilmode='RGB').astype(np.float)
        # if img.shape[0] == img.shape[1] and img.shape[0] >= size:
        img_resized = skimage.transform.resize(img, (size, size))
        file_name = os.path.basename(image_file)
        imageio.imwrite("./datasets/set14/" + file_name, img_resized)
        count += 1
        if count == 1000:
            break


###get an image, down sampling 4x, and plot it###
def downsample_plot(img_file):
    size = 512
    img_name = os.path.splitext(img_file)[0] 
    h = size
    w = size
    low_h, low_w = int(h / 4), int(w / 4)

    img = imageio.imread(img_name + '.png', pilmode='RGB').astype(np.float)

    img_lr = skimage.transform.resize(img, (low_h, low_w))
    imageio.imwrite(img_name + '_lr.png', img_lr)

    # img=mpimg.imread('188120.jpg')
    # print(img)

    fig = plt.figure()
    plt.imshow(img_lr.astype(np.int))
    fig.savefig(img_name + '_lr_plot.png')
    plt.show()
    plt.close()

    # get a small image, and resize 4x upsampling, bicubic, etc
    sess = tf.InteractiveSession()

    with open(img_name + '_lr.png', 'rb') as f:
        image_bytes = f.read()

    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.expand_dims(image, 0)

    # resized_image = tf.image.resize_bicubic(image, [400, 400])
    # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/resize_images?hl=in
    resized_image = tf.image.resize_images(image, [size, size], method=tf.image.ResizeMethod.BILINEAR)

    resized_image = tf.cast(resized_image, tf.uint8)
    resized_image = tf.squeeze(resized_image)
    encoded_image = tf.image.encode_jpeg(resized_image)

    print(tf.shape(encoded_image).eval())

    # with tf.Session() as sess:
    jpg_image = sess.run(encoded_image)
    with open(img_name + '_bilinear_sr.png', 'wb') as f:
        f.write(jpg_image)


def compute_psnr(img_file):
    img_name = os.path.splitext(img_file)[0]
    sess = tf.InteractiveSession()

    with open(img_file, 'rb') as f:
        image_bytes1 = f.read()

    #with open(img_name + '8250_000000149739_fakehr1.png', 'rb') as f:
    with open('8250_000000149739_fakehr1.png', 'rb') as f:
        image_bytes2 = f.read()

    #with open(img_name + '000000149739_4x.jpg', 'rb') as f:
    with open('000000149739_4x.jpg', 'rb') as f:
        image_bytes3 = f.read()

    im1 = tf.image.decode_png(image_bytes1, channels=3)
    im2 = tf.image.decode_png(image_bytes2, channels=3)
    im3 = tf.image.decode_png(image_bytes3, channels=3)

    psnr_sr = tf.image.psnr(im1, im2, max_val=255)
    psnr_bilinear = tf.image.psnr(im1, im3, max_val=255)
    print(psnr_sr.eval())
    print(psnr_bilinear.eval())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image processing')
    parser.add_argument('--mode', default='', help='running mode')
    parser.add_argument('--file', default='', help='image file to process')

    args = parser.parse_args()
    img_file = args.file

    if args.mode == 'downsample_plot' and img_file:
        downsample_plot(img_file)
    elif args.mode == 'produce_training_data':
        produce_training_data()
    elif args.mode == 'psnr' and img_file:
        compute_psnr(img_file)
    else:
        print("--mode should be downsample_plot/produce_training_data/psnr")
