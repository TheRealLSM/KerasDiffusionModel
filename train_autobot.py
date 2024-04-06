import tensorflow as tf
import glob
import numpy as np


def define_model(width, height):
    encode_layers = decode_layers = 150
    dropout_level = 0.1
    kernel_size = (5, 5)
    strides = (2, 2)
    activation_function = 'tanh'

    model_image_input = tf.keras.layers.Input(shape=(width, height, 3), name='image_input')

    model_image_decomp3 = tf.keras.layers.Conv2D(encode_layers, kernel_size=(5, 5), strides=strides, padding='valid', activation=activation_function)(model_image_input)
    model_image_decomp3b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_image_decomp3)
    model_image_decomp3c = tf.keras.layers.Dropout(dropout_level)(model_image_decomp3b)

    model_image_decomp4 = tf.keras.layers.Conv2D(encode_layers*2, kernel_size=kernel_size, strides=strides, padding='valid', activation=activation_function)(model_image_decomp3c)
    model_image_decomp4b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_image_decomp4)
    model_image_decomp4c = tf.keras.layers.Dropout(dropout_level)(model_image_decomp4b)

    model_image_decomp5 = tf.keras.layers.Conv2D(encode_layers*4, kernel_size=kernel_size, strides=strides, padding='valid', activation=activation_function)(model_image_decomp4c)
    model_image_decomp5b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_image_decomp5)
    model_image_decomp5c = tf.keras.layers.Dropout(dropout_level)(model_image_decomp5b)

    model_image_decomp6 = tf.keras.layers.Conv2D(encode_layers*8, kernel_size=kernel_size, strides=strides, padding='valid', activation=activation_function)(model_image_decomp5c)
    model_image_decomp6b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_image_decomp6)
    model_image_decomp6c = tf.keras.layers.Dropout(dropout_level)(model_image_decomp6b)

    model_image_decomp10 = tf.keras.layers.Conv2D(encode_layers*16, kernel_size=kernel_size, strides=strides, padding='valid', activation=activation_function)(model_image_decomp6c)
    model_image_decomp10b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_image_decomp10)
    model_image_decomp10c = tf.keras.layers.Dropout(dropout_level)(model_image_decomp10b)

    model_image_decomp11 = tf.keras.layers.Conv2D(encode_layers*16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=activation_function)(model_image_decomp10c)
    model_image_decomp11b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_image_decomp11)
    model_image_decomp11c = tf.keras.layers.Dropout(dropout_level)(model_image_decomp11b)

    model_step_input = tf.keras.layers.Input(shape=1, name='step_input')
    model_step_input2 = tf.keras.layers.Reshape(skip_con_shape)(model_step_input)

    model_decoder1 = tf.keras.layers.concatenate([model_image_decomp11c, model_step_input2], axis=-1)
    model_decoder1b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_decoder1)

    model_decoder2 = tf.keras.layers.Conv2DTranspose(decode_layers*8, kernel_size=kernel_size, strides=strides, padding='valid', activation=activation_function)(model_decoder1b)
    model_decoder2b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_decoder2)
    model_decoder2c = tf.keras.layers.Dropout(dropout_level)(model_decoder2b)

    model_decoder4 = tf.keras.layers.Conv2DTranspose(decode_layers*4, kernel_size=kernel_size, strides=strides, padding='valid', activation=activation_function)(model_decoder2c)
    model_decoder4b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_decoder4)
    model_decoder4c = tf.keras.layers.Dropout(dropout_level)(model_decoder4b)

    model_decoder5 = tf.keras.layers.Conv2DTranspose(decode_layers*2, kernel_size=kernel_size, strides=strides, padding='valid',
                                                 activation=activation_function)(model_decoder4c)
    model_decoder5b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_decoder5)
    model_decoder5c = tf.keras.layers.Dropout(dropout_level)(model_decoder5b)

    model_decoder7 = tf.keras.layers.Conv2DTranspose(decode_layers, kernel_size=kernel_size, strides=strides, padding='valid',
                                                     activation=activation_function)(model_decoder5c)
    model_decoder7b = tf.keras.layers.BatchNormalization(momentum=batchnorm_momentum)(model_decoder7)
    model_decoder7c = tf.keras.layers.Dropout(dropout_level)(model_decoder7b)

    model_out = tf.keras.layers.Conv2DTranspose(3, kernel_size=kernel_size, strides=strides, padding='valid', activation='sigmoid')(model_decoder7c)

    model = tf.keras.models.Model([model_image_input, model_step_input],  model_out)
    optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.0001,
    weight_decay=0.005,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07)
    model.compile(optimizer=optimizer, loss='MeanSquaredError')

    return model

width = height = output_dim = 125
batchnorm_momentum = 0.99
batch_number = 1
num_noise_turns = 10
skip_con_shape = (1, 1, 1)

print('Fetching file names')
path = "./images/**/*.jp*g"
file_names = []

for path in glob.glob(path, recursive=True):
    file_names.append(path)

#model = define_model(width, height)
model = tf.keras.models.load_model('difusion_model_small_output.keras')
model.summary()

print('Model training begins')

while 0 != 1:
    # For every 1000 batches trained test progress and save results as well as the model
    if batch_number % 1000 == 0:
        print('Batch number ' + str(batch_number))
        print('Generating image')
        img = np.random.normal(0, 15, (1, width, height, 3))

        # Check a full denoising pass to measure success
        for i in range(0, num_noise_turns):
            X2 = np.full(1, (num_noise_turns - 1 - i) / num_noise_turns)
            img = np.interp(img, (img.min(), img.max()), (0.03, 0.95)).reshape((1, width, height, 3))
            img = model([img, X2], training=False).numpy()[0]
            img = np.interp(img, (img.min(), img.max()), (0.03, 0.95)).reshape((1, width, height, 3))
            tf.keras.utils.save_img('./output_results/test_image_' + str(batch_number) + '_' + str(i) + '.jpg', (np.asarray(img[0]) * 256).astype(np.uint8))

        # Sanity check with a noisy image that model works
        img = tf.image.resize(tf.keras.utils.load_img(file_names[file_number]), size=[width, height],
                              antialias=True) / 256
        noise_map = np.random.normal(0, 0.2, (width, height, 3))
        noisy_image = img + noise_map
        noisy_image = np.interp(noisy_image, (noisy_image.numpy().min(), noisy_image.numpy().max()), (0, 1))
        tf.keras.utils.save_img('denoising_test_input' + str(x) + '.jpg', (np.asarray(noisy_image) * 256).astype(np.uint8))
        X1 = np.zeros((1, width, height, 3))
        X1[0] = noisy_image
        X2 = np.full((1, 1), 0.5)
        img = (model([X1, X2], training=False).numpy()[0])
        tf.keras.utils.save_img('denoising_test_result' + str(x) + '.jpg', (np.asarray(img) * 256).astype(np.uint8))

        # Save model
        model.save('difusion_model_small_output.keras')

    # Fetch a new random file number
    file_number = np.random.randint(len(file_names))

    # Initialize input and output matrices
    X1 = np.zeros((num_noise_turns, width, height, 3))
    X2 = np.zeros(num_noise_turns)
    Y = np.zeros((num_noise_turns, width, height, 3))

    # Load image in 0 to 1 pixel value format
    img = (tf.keras.utils.img_to_array(tf.image.resize(tf.keras.utils.load_img(file_names[file_number]), size=[width, height], antialias=True)).astype(np.float32) / 255)
    min_val = np.min(img)
    max_val = np.max(img)
    noisy_image = img
    noise_map = np.zeros((width, height, 3))

    for x in range(0, num_noise_turns):
        noise_map = noise_map + np.random.normal(0, 0.019, (width, height, 3))
        try:
            noisy_image = img + noise_map
            noisy_image = np.interp(noisy_image, (noisy_image.min(), noisy_image.max()), (min_val, max_val))
            X1[x] = noisy_image
            X2[x] = x / num_noise_turns
            Y[x] = img
        except ValueError:
            print('Problem file: ' + str(file_names[file_number]))
            exit()

        img = X1[x]
#    exit()

    model.train_on_batch([X1, X2], Y)

    batch_number += 1

print('We will nevewr reach this as the loop above continues indefinitely.')
