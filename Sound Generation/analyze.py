import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tns
import os
from PIL import Image
from autoEncoder import AutoEncoder, VarEncoder

def load_fsdd(path):
    x = []
    for root, _, names in os.walk(path):
        for filename in names:
            filepath = os.path.join(root, filename)
            spectro = np.load(filepath)
            x.append(spectro)
    x = np.array(x)
    x = x[..., np.newaxis]
    return x

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels

def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()

def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
   # autoencoder = AutoEncoder.load("encoderFULL")
   # varencoder = VarEncoder.load("hyperAccurate")
    x_train = load_fsdd("spectrograms")
    autoencoder = VarEncoder((256,64,1), (512,256,128,64,32), (3,3,3,3,3), (2,2,2,2, (2,1)), 128)
    autoencoder.compile(0.0001)
    autoencoder.train(x_train=x_train, batch_size=32,num_epochs=20)
    autoencoder.save("spectroreader")
