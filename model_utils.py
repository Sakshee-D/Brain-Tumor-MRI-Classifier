from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_ROOT / "models"
DEFAULT_IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class DatasetError(RuntimeError):
    pass


def import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
        from tensorflow.keras.metrics import Precision, Recall
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install requirements.txt before training or prediction."
        ) from exc

    return {
        "tf": tf,
        "Adam": Adam,
        "Conv2D": Conv2D,
        "Dense": Dense,
        "Dropout": Dropout,
        "Flatten": Flatten,
        "ImageDataGenerator": ImageDataGenerator,
        "Input": Input,
        "MaxPooling2D": MaxPooling2D,
        "Precision": Precision,
        "Recall": Recall,
        "Sequential": Sequential,
    }


def safe_extract(zip_file, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        for child in destination.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    with zipfile.ZipFile(zip_file) as archive:
        target_root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target_root != target and target_root not in target.parents:
                raise DatasetError("The uploaded ZIP contains an unsafe path.")
        archive.extractall(destination)

    return destination


def extract_dataset_zip(uploaded_file, destination: Path) -> Path:
    if uploaded_file is None:
        raise DatasetError("Please upload a dataset ZIP file.")
    return safe_extract(uploaded_file, destination)


def _contains_images(path: Path) -> bool:
    return any(file.suffix.lower() in IMAGE_EXTENSIONS for file in path.rglob("*") if file.is_file())


def _class_dirs(path: Path) -> list[Path]:
    return [child for child in path.iterdir() if child.is_dir() and _contains_images(child)]


def find_image_dataset_dir(root: Path, preferred_name: str | None = None) -> Path:
    candidates = []
    if preferred_name:
        candidates.extend([p for p in root.rglob(preferred_name) if p.is_dir()])
    candidates.append(root)
    candidates.extend([p for p in root.rglob("*") if p.is_dir()])

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if len(_class_dirs(candidate)) >= 2:
            return candidate

    raise DatasetError(
        "Could not find a class-folder image dataset. Expected folders like "
        "Training/glioma, Training/meningioma, Training/notumor, Training/pituitary."
    )


def get_class_names(train_dir: Path) -> list[str]:
    class_names = sorted(child.name for child in train_dir.iterdir() if child.is_dir() and _contains_images(child))
    if len(class_names) < 2:
        raise DatasetError("Training data must contain at least two class folders with images.")
    return class_names


def build_model(num_classes: int, image_size: tuple[int, int], learning_rate: float):
    keras = import_tensorflow()
    model = keras["Sequential"](
        [
            keras["Input"](shape=(image_size[0], image_size[1], 3)),
            keras["Conv2D"](32, (3, 3), activation="relu"),
            keras["MaxPooling2D"]((2, 2)),
            keras["Conv2D"](64, (3, 3), activation="relu"),
            keras["MaxPooling2D"]((2, 2)),
            keras["Conv2D"](128, (3, 3), activation="relu"),
            keras["MaxPooling2D"]((2, 2)),
            keras["Flatten"](),
            keras["Dropout"](0.5),
            keras["Dense"](512, activation="relu"),
            keras["Dense"](num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras["Adam"](learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras["Precision"](name="precision"), keras["Recall"](name="recall")],
    )
    return model


def train_model(
    train_dir: Path,
    image_size: tuple[int, int],
    batch_size: int,
    epochs: int,
    learning_rate: float,
    validation_split: float,
    epoch_callback: Callable[[int, int, dict[str, float]], None] | None = None,
):
    keras = import_tensorflow()
    class_names = get_class_names(train_dir)

    datagen = keras["ImageDataGenerator"](
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        validation_split=validation_split,
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        classes=class_names,
    )
    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        classes=class_names,
    )

    callbacks = []
    if epoch_callback:
        class StreamlitProgressCallback(keras["tf"].keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                epoch_callback(epoch + 1, epochs, logs or {})

        callbacks.append(StreamlitProgressCallback())

    model = build_model(len(class_names), image_size, learning_rate)
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
    )
    return model, class_names, history


def evaluate_model(model, test_dir: Path, class_names: list[str], image_size: tuple[int, int], batch_size: int):
    keras = import_tensorflow()
    from sklearn.metrics import classification_report, confusion_matrix

    test_datagen = keras["ImageDataGenerator"](rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        classes=class_names,
    )
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    report_dict = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report = pd.DataFrame(report_dict).transpose()
    matrix = pd.DataFrame(confusion_matrix(true_classes, predicted_classes), index=class_names, columns=class_names)
    return report, matrix


def save_artifacts(model, class_names: list[str], history: dict[str, list[float]], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "tumor_cnn.keras")
    (model_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (model_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_trained_model(model_dir: Path):
    keras = import_tensorflow()
    model_path = model_dir / "tumor_cnn.keras"
    labels_path = model_dir / "class_names.json"
    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Missing model artifacts. Train the model or add files under models/.")
    model = keras["tf"].keras.models.load_model(model_path)
    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    return model, class_names


def preprocess_image(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    resized = image.convert("RGB").resize(image_size)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(model, image: Image.Image, class_names: list[str], image_size: tuple[int, int]) -> list[tuple[str, float]]:
    predictions = model.predict(preprocess_image(image, image_size), verbose=0)[0]
    ranked = sorted(zip(class_names, predictions), key=lambda item: float(item[1]), reverse=True)
    return [(label, float(score)) for label, score in ranked]
