import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset import Caltech101Dataset
from utils.train_utils import train_one_epoch, validate
from models.resnet_finetune import get_finetuned_resnet
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--train_split', type=str, required=True)
    parser.add_argument('--val_split', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--logdir', type=str, default='logs', help='Path to TensorBoard log directory')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_dataset = Caltech101Dataset(args.data_root, args.train_split, train_transforms)
    val_dataset = Caltech101Dataset(args.data_root, args.val_split, val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_finetuned_resnet(num_classes=101, pretrained=args.pretrained).to(device)

    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = True

    import torch.optim as optim
    optimizer = optim.SGD([
        {'params': [p for n,p in model.named_parameters() if 'fc' in n], 'lr': args.lr},
        {'params': [p for n,p in model.named_parameters() if 'fc' not in n], 'lr': args.finetune_lr}
    ], momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=args.logdir)
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
    writer.close()

if __name__ == '__main__':
    main()
