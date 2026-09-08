import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Train the waste detector/segmenter")
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--model", default="yolo11n-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    model = YOLO(args.model)
    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        pretrained=True,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        degrees=10,
        translate=0.10,
        scale=0.50,
        shear=2,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        project="runs/waste",
        name="waste_detector",
    )
    if args.device:
        train_args["device"] = args.device
    model.train(**train_args)
    print("Training complete. Check runs/waste/waste_detector/weights/best.pt")


if __name__ == "__main__":
    main()
