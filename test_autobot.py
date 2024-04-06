import tensorflow as tf
from PIL import Image, ImageFilter
import glob
import numpy as np


width = height = output_dim = 125
batchnorm_momentum = 0.99
img_x = 8
img_y = 4
images = []

model = tf.keras.models.load_model('difusion_model_small_output.keras')
model.summary()

print('Start generating samples')
num_noise_turns = 10

for run_number in range(0, 8*4):
    val_min = np.random.rand() * 0.1
    val_max = 1 - np.random.rand() * 0.1

    X1 = np.random.normal(0, 1, (1, width, height, 3))
    X1 = np.interp(X1, (X1.min(), X1.max()), (val_min, val_max))

    for i in range(0, num_noise_turns):
        X2 = np.full(1, (num_noise_turns - 1 - i) / num_noise_turns)
        X1 = model([X1, X2], training=False).numpy()
        img = Image.fromarray((X1[0] * 255).astype(np.uint8)).resize((250, 200))
        if i == num_noise_turns - 1:
            images.append(img)

background = Image.new("RGB", (250*img_x+100, 200*img_y+100), (255, 255, 255))

for y in range(0, img_y):
    for x in range(0, img_x):
        background.paste(images[x+y*img_x], (x*250+50, y*200+50))


background.filter(ImageFilter.SHARPEN).save('./output_results/model_output_test.jpg')
