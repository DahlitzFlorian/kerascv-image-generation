import time

import keras_cv
import matplotlib.pyplot as plt

from tensorflow import keras

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

t = time.time()
images = model.text_to_image(
    "cute magical flying dog, fantasy art, "
    "golden color, high quality, highly detailed, elegant, sharp focus, "
    "concept art, character concepts, digital painting, mystery, adventure",
    batch_size=3,
)
t = time.time() - t
print(f"t: {t} sec.")


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    
    plt.savefig("img.png", bbox_inches='tight')


plot_images(images)
