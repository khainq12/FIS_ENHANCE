from datasets import load_dataset

ds = load_dataset("korexyz/celeba-hq-256x256")

print(ds)
print("Train images:", len(ds["train"]))
print("Validation images:", len(ds["validation"]))

img = ds["train"][0]["image"]
print("Image size:", img.size)