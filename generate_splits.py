import os
import random
from pathlib import Path

random.seed(42)

def generate_splits(data_root='data/caltech-101', train_per_class=30, val_per_class=5):
    all_classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d != 'BACKGROUND_Google'])

    label_map = {cls_name: idx for idx, cls_name in enumerate(all_classes)}

    train_lines, val_lines, test_lines = [], [], []

    for cls in all_classes:
        cls_dir = Path(data_root) / cls
        images = sorted(cls_dir.glob("*.jpg"))
        images = [str(img.relative_to(data_root)) for img in images]
        random.shuffle(images)

        label = label_map[cls]

        train_imgs = images[:train_per_class]
        val_imgs = train_imgs[:val_per_class]
        train_imgs = train_imgs[val_per_class:]

        test_imgs = images[train_per_class:]

        train_lines += [f"{img} {label}\n" for img in train_imgs]
        val_lines += [f"{img} {label}\n" for img in val_imgs]
        test_lines += [f"{img} {label}\n" for img in test_imgs]

    os.makedirs("splits", exist_ok=True)
    with open("splits/train.txt", "w") as f:
        f.writelines(train_lines)
    with open("splits/val.txt", "w") as f:
        f.writelines(val_lines)
    with open("splits/test.txt", "w") as f:
        f.writelines(test_lines)

    print("✅ Split files generated: splits/train.txt, splits/val.txt, splits/test.txt")

if __name__ == "__main__":
    generate_splits()
