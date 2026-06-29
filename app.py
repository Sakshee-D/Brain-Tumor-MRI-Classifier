from __future__ import annotations
import os
from huggingface_hub import hf_hub_download
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from model_utils import (
    APP_ROOT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_DIR,
    DatasetError,
    evaluate_model,
    extract_dataset_zip,
    find_image_dataset_dir,
    load_trained_model,
    predict_image,
    save_artifacts,
    train_model,
)


st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    layout="wide",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.4rem;
            max-width: 1180px;
        }
        .metric-card {
            border: 1px solid #d8dee9;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #ffffff;
        }
        .muted {
            color: #5f6b7a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def cached_model(model_dir: str):
    return load_trained_model(Path(model_dir))


def model_status() -> tuple[bool, list[str]]:
    model_path = DEFAULT_MODEL_DIR / "tumor_cnn.keras"
    labels_path = DEFAULT_MODEL_DIR / "class_names.json"
    missing = []
    if not model_path.exists():
        missing.append(str(model_path.relative_to(APP_ROOT)))
    if not labels_path.exists():
        missing.append(str(labels_path.relative_to(APP_ROOT)))
    return not missing, missing


def render_predict() -> None:
    st.title("Brain Tumor MRI Classifier - Using Trained Model")
    st.caption("Upload a brain MRI image and classify it with the trained CNN from the notebook.")

    ready, missing = model_status()
    if not ready:
        st.warning(
            "No trained model artifact was found yet. Train a model in the Train page, "
            f"or deploy with these files present: {', '.join(missing)}."
        )
        return

    with st.spinner("Loading model..."):
        model, class_names = cached_model(str(DEFAULT_MODEL_DIR))

    left, right = st.columns([0.9, 1.1], vertical_alignment="top")
    with left:
        uploaded_image = st.file_uploader(
            "MRI image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=False,
        )

        if uploaded_image:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded MRI", use_container_width=True)
        else:
            st.info("Choose an MRI image to run prediction.")
            return

    with right:
        probabilities = predict_image(model, image, class_names, DEFAULT_IMAGE_SIZE)
        top_label, top_confidence = probabilities[0]

        st.subheader(top_label)
        st.metric("Confidence", f"{top_confidence * 100:.2f}%")
        st.progress(float(top_confidence))

        st.write("Class probabilities")
        for label, score in probabilities:
            st.write(f"{label}: {score * 100:.2f}%")
            st.progress(float(score))


def render_train() -> None:
    st.title("Train and Evaluate on your Custom Dataset")
    st.caption("Train the notebook CNN from uploaded ZIPs or the local dataset folders under dataset/.")

    local_train_dir = APP_ROOT / "dataset" / "Training"
    local_test_dir = APP_ROOT / "dataset" / "Testing"

    if local_train_dir.exists() and local_test_dir.exists():
        st.info(f"Detected local datasets at {local_train_dir} and {local_test_dir}.")
    else:
        st.info("Place training and testing images under dataset/Training and dataset/Testing, or upload ZIP archives.")

    train_zip = st.file_uploader("Training.zip", type=["zip"], key="train_zip")
    test_zip = st.file_uploader("Testing.zip", type=["zip"], key="test_zip")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20, step=1)
    with col_b:
        batch_size = st.number_input("Batch size", min_value=4, max_value=128, value=32, step=4)
    with col_c:
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.00001,
            max_value=0.1,
            value=0.001,
            step=0.0001,
            format="%.5f",
        )

    validation_split = st.slider("Validation split", 0.05, 0.40, 0.20, 0.05)

    can_train = (train_zip is not None and test_zip is not None) or (
        local_train_dir.exists() and local_test_dir.exists()
    )

    if not st.button("Train model", type="primary", disabled=not can_train):
        return

    try:
        if train_zip is not None and test_zip is not None:
            with st.spinner("Extracting uploaded datasets..."):
                train_extract = extract_dataset_zip(train_zip, APP_ROOT / "data" / "training_upload")
                test_extract = extract_dataset_zip(test_zip, APP_ROOT / "data" / "testing_upload")
                train_dir = find_image_dataset_dir(train_extract, preferred_name="Training")
                test_dir = find_image_dataset_dir(test_extract, preferred_name="Testing")
        else:
            with st.spinner("Preparing local datasets..."):
                train_dir = find_image_dataset_dir(local_train_dir, preferred_name="Training")
                test_dir = find_image_dataset_dir(local_test_dir, preferred_name="Testing")

        st.success(f"Using training data at `{train_dir}` and testing data at `{test_dir}`.")

        progress = st.progress(0.0)
        status = st.empty()

        def on_epoch(epoch: int, total: int, logs: dict[str, float]) -> None:
            progress.progress(epoch / total)
            accuracy = logs.get("accuracy", 0.0)
            val_accuracy = logs.get("val_accuracy", 0.0)
            status.write(
                f"Epoch {epoch}/{total} - accuracy {accuracy:.4f}, validation accuracy {val_accuracy:.4f}"
            )

        model, class_names, history = train_model(
            train_dir=train_dir,
            image_size=DEFAULT_IMAGE_SIZE,
            batch_size=int(batch_size),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            validation_split=float(validation_split),
            epoch_callback=on_epoch,
        )

        with st.spinner("Evaluating test set..."):
            report, confusion = evaluate_model(
                model=model,
                test_dir=test_dir,
                class_names=class_names,
                image_size=DEFAULT_IMAGE_SIZE,
                batch_size=int(batch_size),
            )
            save_artifacts(model, class_names, history.history, DEFAULT_MODEL_DIR)
            cached_model.clear()

        st.success("Model trained and saved to `models/tumor_cnn.keras`.")

        st.subheader("Evaluation")
        st.dataframe(report, use_container_width=True)
        st.write("Confusion matrix")
        st.dataframe(confusion, use_container_width=True)

    except DatasetError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)


def main() -> None:
    _inject_styles()
    page = st.sidebar.radio("Page", ["Predict", "Train"], label_visibility="collapsed")

    if page == "Predict":
        render_predict()
    else:
        render_train()


if __name__ == "__main__":
    main()
