import matplotlib.pyplot as plt


# for training steps generate the images
def generate_image(model, original, transformed, step=None):
    # generated_images
    generated_image = model(original, training=True)

    # plot the images
    plt.figure(figsize=(12, 8))

    img_list = [original[0], transformed[0], generated_image[0]]

    title_list = ["Orjinal Resim", "Beklenen Transform", "Generator Resim"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title_list[i])
        plt.imshow(img_list[i] * 0.5 + 0.5)
        plt.axis("off")

    if step is not None:
        plt.savefig(f"created_images_cgan/{step+1}_generated.png", bbox_inches="tight")