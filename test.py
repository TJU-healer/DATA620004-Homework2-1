import argparse
import torch
from torchvision import transforms
from utils.dataset import Caltech101Dataset
from models.resnet_finetune import get_finetuned_resnet
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--test_split', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    test_dataset = Caltech101Dataset(args.data_root, args.test_split, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = get_finetuned_resnet(num_classes=101, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == '__main__':
    main()
