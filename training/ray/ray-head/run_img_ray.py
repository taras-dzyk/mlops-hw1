import wandb
import tensorflow as tf
import numpy as np
import os
import shutil
from PIL import Image
from pathlib import Path
import ray
import tempfile

wandb.login(key=os.environ.get("WANDB_API_KEY"))
api = wandb.Api()

artifact = api.artifact("taras-dzyk-personal/ray-img-demo/crockodile-finder-model-128:latest", type="model")
artifact_dir = artifact.download()

print("Model downloaded to:", artifact_dir)

model_path = os.path.join(artifact_dir, "model_128.keras")
print("Full model path:", model_path)


def collect_images(input_dir="input"):
    input_dir = Path(input_dir)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    def is_image_file(path):
        return path.suffix.lower() in image_extensions

    return [img for img in input_dir.glob("*") if img.is_file() and is_image_file(img)]


def load_and_preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    return img_array


@ray.remote
class Predictor:
    def __init__(self, model_bytes):
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(model_bytes)
            self.temp_model_path = tmp.name
        self.model = tf.keras.models.load_model(self.temp_model_path)

    def predict(self, images_batch, filenames_batch, probability=0.5):
        matched = []
        imgs_tensor = tf.convert_to_tensor(images_batch)
        preds = self.model.predict(imgs_tensor, verbose=0)
        for filename, pred in zip(filenames_batch, preds):
            score = float(pred[0])
            if score > probability:
                matched.append((filename, score))
        return matched

def copy_matched_images(matched, output_dir="output", input_dir="input"):
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    output_dir.mkdir(exist_ok=True)

    for filename, score in matched:
        shutil.copy(input_dir / filename, output_dir / filename)
        printg(f"Знайдено крокодила! {filename} (score: {score:.3f})")


def printv(text): print(f"\033[95m{text}\033[0m")
def printg(text): print(f"\033[92m{text}\033[0m")
def printr(text): print(f"\033[91m{text}\033[0m")
def printy(text): print(f"\033[93m{text}\033[0m")


def main_inference(model_path, probability=0.5, batch_size=8, input_dir="input"):
    
    ray.init(runtime_env={"working_dir": artifact_dir})

    image_paths = collect_images(input_dir)
    printy(f"Знайдено {len(image_paths)} зображень")


    with open(model_path, "rb") as f:
        model_bytes = f.read()
    model_bytes_ref = ray.put(model_bytes)


    # Підготовка батчів із масивів та імен файлів
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [load_and_preprocess(p) for p in batch_paths]
        filenames = [p.name for p in batch_paths]
        batches.append((images, filenames))

    predictor = Predictor.remote(model_bytes_ref)

    futures = [predictor.predict.remote(images, filenames, probability=probability) for images, filenames in batches]

    results = ray.get(futures)
    matched = [item for sublist in results for item in sublist]

    printg(f"\nКрокодилів знайдено: {len(matched)}")
    copy_matched_images(matched, input_dir=input_dir)



    
main_inference(model_path=model_path, probability=0.5, batch_size=8)
