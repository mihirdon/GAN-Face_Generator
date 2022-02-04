import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import RMSprop
import time
import IPython
from IPython import display
from sklearn.preprocessing import StandardScaler

import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image

FOLDER_FACES = '/Users/mihir/downloads/GAN_Face_Folder/img_celebs' # CHANGE DEPENDING ON USER ----- In the case I send this to bestie arnav
NUM_OF_IMAGES = 10000
crop_param = (0, 15, 178, 208 - 15) #hardcoded to see what makes it look good centering around the face: order is top, left, bottom, right
BATCH_SIZE = 8

image_list = []

for pic_file_name in (os.listdir(FOLDER_FACES)[:NUM_OF_IMAGES]): #for every image in (download Folder of pictures of faces) up to NUM_OF_IMAGES images
    pic = Image.open(FOLDER_FACES +  '/' + pic_file_name).crop(crop_param) #crops image to 128 + lots of buffer x 128 + lots of buffer (buffer to make sure we have space to include all parts of face)
    pic.thumbnail((128, 128), Image.ANTIALIAS) #refocuses image to center around main parts of face
    image_list.append(np.uint8(pic)) #makes the pictures into a number so it is more easily storable


image_list = np.array(image_list)
image_list.reshape(image_list.shape[0], 128, 128, 3).astype('float32')
image_list = (image_list - 127.5) / 127.5
print(image_list.shape)
train_dataset = tf.data.Dataset.from_tensor_slices(image_list).shuffle(NUM_OF_IMAGES).batch(BATCH_SIZE)


#display 25 images just to see what our beautiful beautiful inputs look like when actually running the program can ignore this
plt.figure(1, figsize=(10,10))
#for i in range(25):
    #plt.subplot(5, 5, i+1)
    #plt.imshow(image_list[i])
    #plt.axis('off')
#plt.show()


INPUT_VEC_SIZE = 32
COLOR_CHANNELS = 3


#Guide To Layer Stuff I Learned:
#   - Input Layer: Basic input layer has the different shit that does the thing you understand
#   - Nonlinearization Layer: the non linear activation function, in this case relu (Leaky RELU)
#   - Convolutional Layer: THIS is our important layers, they allow for the understanding of lines when stacked,
#                           and also allow for other layers to understand basic shapes and shit very cool very cool (through feature maps
#                           basically isolates our different parts of the image)
#   - Pooling Layer: ALSO our important layers, operate on the feature map, typically in small squares with gaps inbetween
#                    so it will look at four pixels lets say then another four pixels in a square than skip a few etc.
#                    cuts our dimensions in half every time, great for getting to the 128 x 128 we desire


# Vanishing Gradient Problem: 
#       With a large enough model, the backpropogation that normally comes from the gradient stuff will become less and less as it goes backward.
#   That's just an aspect of backpropogation. But the problem that comes with this is that if the backpropogation backpropogates far enough then 
#   it becomes SOOOO small that the parameters in the back simply do not change and this can lead to some funky stuff happening. That is why for 
#   our model we use LeakyReLU cuz it helps deal with that problem because of the way its designed. I suggest looking up the specifics.
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16 * 16 * 128, use_bias=False, input_shape=(INPUT_VEC_SIZE,))) #Basic input layer
    model.add(layers.LeakyReLU()) #Nonlinearization layer
    print(model.output_shape)  # Note: None is the batch size

    model.add(layers.Conv2D(256, 5, padding='same')) #Convolutional
    model.add(layers.LeakyReLU()) #Nonlinear, ALSO LeakyReLU helps with vanishing gradient problem!!! Fun and quirky of it
    print(model.output_shape)

    model.add(layers.Conv2DTranspose(256, 2, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    print(model.output_shape)

    model.add(layers.Conv2DTranspose(256, 2, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    print(model.output_shape)

    model.add(layers.Conv2DTranspose(256, 2, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    print(model.output_shape)

    model.add(layers.Conv2D(512, 7, padding='same'))  #Pooling
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(512, 7, padding='same'))  #Pooling 
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(512, 7, padding='same'))  #Pooling 
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(COLOR_CHANNELS, 1, activation='tanh')) #Pooling but since our whole thing was designed for tahn we do be using tahn
    print(model.output_shape)

    return model #messed with it till it got to a point where I liked how the random generator looks (the image produced below as well as getting 
    #             it to produce a 128 x 128 image)

generator = make_generator_model()

noise = tf.random.normal([1, 32])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])
plt.show() #used for when wanting to display the image (originally created for testing things out) should be random color pixels


# Making the Discriminator Model ___________________________________________________________________________________________________________________

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(256, 3, input_shape=[128, 128, 3])) #input layer and also apply convolution to set up the line stuff I talked about before
    model.add(layers.LeakyReLU()) 

    model.add(layers.Conv2D(256, 4, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1)) #drop 1/10 of the data we have for character development (in the same way we scrape out some of a starter to make it strong)

    model.add(layers.Conv2D(256, 4, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(256, 4, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(256, 4, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='tanh')) #consider changing activation function in the case it don turn out great
    
    optimizer = RMSprop(
        learning_rate=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(discriminator)
print(decision) #positive numbers indicate real image, negative indicates fake image

# Creating The Loss Functions _____________________________________________________________________________________________________________________

optimizer = RMSprop(
        learning_rate=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

#discriminator.trainable = False #depeding on image quality remove this ??? (in online guides it says put this but in the first tutorial they did not)

gan_input = tf.keras.Input(shape = (INPUT_VEC_SIZE, ))
gan_output = discriminator(generator(gan_input))

gan = tf.keras.Model(gan_input,gan_output)

gan.compile(optimizer = optimizer, loss = 'binary_crossentropy')

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


#made them seperate just in case so when called in the future they don't impact the other one
gen_opt = RMSprop(
        learning_rate=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

disc_opt = RMSprop(
        learning_rate=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

print("finished loss function creation")
# Training Loop _______________________________________________________________________________________________________________----__--__________-


#checkpoints set up
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                 discriminator_optimizer=disc_opt,
                                 generator=generator,
                                 discriminator=discriminator)


EPOCHS = 20000
NOISE_DIM = 32
NUM_EXAMPLES_TO_GENERATE = 36 # when we print out the images we want them in a 6 x 6 format
images_saved = 0

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

RES_DIR = FOLDER_FACES + "/res2"
FILE_PATH = "%sgenerated%d.png"

if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)


# Training Step Function ---- Where all the magic actually happens
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


print("started stepping")
# the loop for the train_step function to be called, also saves images for the gif creation
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    print("started epoch")
    for image_batch in dataset:
      train_step(image_batch)
      print("finished training step")
      

    # Produce images for the GIF as you go (every 100 epochs)
    if (epoch % 100 == 0):
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epoch + 1,
                                 seed,
                                 images_saved)

    # Save the model every 1000 epochs
    if (epoch + 1) % 1000 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


# Generating and Saving Images __________________________________________________________________________________________

def generate_and_save_images(model, epoch, test_input, images_saved):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(6, 6)) #output an image that is 6 x 6

  for i in range(predictions.shape[0]):
      plt.subplot(6, 6, i+1)
      plt.imshow(predictions[i, :, :, 0])
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

  fig.save(FILE_PATH % (RES_DIR, images_saved))
  images_saved += 1

# FInally We are done ________________________________________________________________________________________________---

train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
display_image(EPOCHS)
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)

