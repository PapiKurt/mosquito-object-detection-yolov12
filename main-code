import os
import subprocess
import cv2
import numpy as np
from matplotlib import pyplot as plt


# ==========================================================
# INSTALL DEPENDENCIES
# ==========================================================

def install_dependencies():
    """
    Install all required dependencies in one call.
    """

    packages = [
        "torch==2.2.2",
        "torchvision==0.17.2",
        "timm==1.0.14",
        "albumentations==2.0.4",
        "pycocotools==2.0.7",
        "PyYAML==6.0.1",
        "scipy==1.13.0",
        "onnxslim==0.1.31",
        "onnxruntime-gpu==1.18.0",
        "gradio==4.44.1",
        "opencv-python==4.9.0.80",
        "psutil==5.9.8",
        "py-cpuinfo==9.0.0",
        "huggingface-hub==0.23.2",
        "safetensors==0.4.3",
        "numpy==1.26.4",
        "supervision==0.22.0",
        "ultralytics==8.3.176",
        "onnx==1.16.2",
        "flash-attn==2.6.3"
    ]

    print("Installing dependencies...")

    subprocess.run(
        ["pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    )

    subprocess.run(
        ["pip", "install"] + packages
    )

    print("Dependencies installed successfully.")


# Call install_dependencies directly at the beginning of the script execution
# to ensure all necessary packages are available before they are imported.
install_dependencies()

# Now import ultralytics after dependencies are installed
from ultralytics import YOLO


# ==========================================================
# SETUP YOLOv12 REPOSITORY
# ==========================================================

def setup_yolov12(project_path):
    """
    Clone YOLOv12 repository and download pretrained weights.
    """

    repo_path = os.path.join(project_path, "yolov12")

    os.makedirs(project_path, exist_ok=True)
    os.chdir(project_path)

    if not os.path.exists(repo_path):

        print("Cloning YOLOv12 repository...")

        subprocess.run([
            "git",
            "clone",
            "https://github.com/sunsmarterjie/yolov12.git"
        ])

    os.chdir(repo_path)

    if not os.path.exists("yolo12n.pt"):

        print("Downloading pretrained YOLOv12 weights...")

        subprocess.run([
            "wget",
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt"
        ])

    return repo_path


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(repo_path, dataset_yaml):

    """
    Train mosquito detection model.
    """

    model_path = os.path.join(repo_path, "yolo12n.pt")

    model = YOLO(model_path)

    results = model.train(

        data=dataset_yaml,
        epochs=1000,
        patience=15,
        batch=8,
        imgsz=512,
        scale=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        close_mosaic=10,
        device="cpu",
        save=True

    )

    return results


# ==========================================================
# RUN INFERENCE
# ==========================================================

def run_inference(repo_path, image_path):

    """
    Run mosquito detection on test image.
    """

    weights = os.path.join(
        repo_path,
        "runs/detect/train/weights/best.pt"
    )

    model = YOLO(weights)

    results = model(image_path)

    for r in results:

        img = cv2.imread(image_path)

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)

        names = model.names

        for box, score, cls_id in zip(boxes, scores, class_ids):

            x1, y1, x2, y2 = map(int, box)

            label = f"{names[cls_id]} {score:.2f}"

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            cv2.putText(
                img,
                label,
                (x1, max(y1-10,0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

