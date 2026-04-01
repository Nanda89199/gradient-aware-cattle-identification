# =============================================
# 🐄 Cattle Detection – YOLO11 Training & Testing
# Save model as: /kaggle/working/yolo11_cattle4.pt
# =============================================

from ultralytics import YOLO
from matplotlib import pyplot as plt
import yaml
import numpy as np
import os
import shutil

os.environ["ULTRALYTICS_NO_AMP_CHECK"] = "1"


def main():

    # -------------------------------
    # Load model
    # -------------------------------
    print(" Loading YOLO model")
    model = YOLO("yolo11.yaml")   # or "yolo11n.pt"


    # -------------------------------
    # Dataset YAML
    # -------------------------------
    yaml_path = "/kaggle/working/data.yaml"

    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    print("\nDataset Info:")
    print(data_cfg)


    # -------------------------------
    # Train
    # -------------------------------
    print("\n🚀 Training started...")

    results = model.train(

        data=yaml_path,

        project="/kaggle/working",

        name="yolo11_cattle4_temp",   # temporary folder

        epochs=100,

        batch=16,

        imgsz=640,

        device=0,

        workers=2,

        save=True

    )

    print("\n✅ Training completed!")


    # -------------------------------
    # Copy best.pt to required name
    # -------------------------------
    best_src = results.save_dir + "/weights/best.pt"

    best_dst = "/kaggle/working/yolo11_cattle4.pt"

    if os.path.exists(best_src):

        shutil.copy(best_src, best_dst)

        print(f"\n✅ Model saved as: {best_dst}")

    else:

        print("❌ best.pt not found!")
        return


    # -------------------------------
    # Load saved model
    # -------------------------------
    print("\n📦 Loading saved model...")

    best_model = YOLO(best_dst)


    # -------------------------------
    # Evaluate on TEST set
    # -------------------------------
    print("\n📈 Testing model...")

    metrics = best_model.val(

        data=yaml_path,

        split="test",

        imgsz=640,

        device=0

    )


    print("\n📊 TEST RESULTS")

    print(f"Precision: {metrics.box.mp:.4f}")

    print(f"Recall: {metrics.box.mr:.4f}")

    print(f"mAP50: {metrics.box.map50:.4f}")

    print(f"mAP50-95: {metrics.box.map:.4f}")


    # -------------------------------
    # Prediction visualization
    # -------------------------------
    print("\n🖼️ Visualizing predictions...")

    sample_images = [

        "/kaggle/input/cattle4-yolo/test/images/1550_2_jpg.rf.8404750b57b1caf470bfe2773c05d3d6.jpg",

        "/kaggle/input/cattle4-yolo/test/images/76_2_jpg.rf.fdfb3b8717ef9bac2ed8443edf1f354b.jpg"

    ]

    for img_path in sample_images:

        if os.path.exists(img_path):

            results = best_model.predict(

                source=img_path,

                conf=0.25,

                save=False

            )

            img = results[0].plot()

            plt.figure(figsize=(8,8))

            plt.imshow(img)

            plt.axis("off")

            plt.show()

        else:

            print("Image not found:", img_path)


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":

    main()