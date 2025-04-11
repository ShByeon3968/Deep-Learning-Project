import torch
from model import DnCNN
from dataset import get_train_loader, get_val_loader
from loss import SumSquaredError
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from utils import show_denoising_result


if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigma = 25
    epoch = 180

    print("모델 구성")
    model = DnCNN()
    model.train()
    criterion = SumSquaredError()
    if cuda:
        model = model.to(device)

    # 옵티마이저, 스케줄러
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    scheduler = MultiStepLR(optimizer,milestones=[30,60,90],gamma=0.2) # learning rate

    train_loader = get_train_loader()
    val_loader = get_val_loader()

    for ep in range(epoch):
        scheduler.step(ep)
        running_loss = 0.0
        for noisy, clean in tqdm(train_loader):
            noisy, clean = noisy.to(device), noisy.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output,clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epoch}] - Training Loss: {epoch_loss:.4f}")

        # 검증 루프
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in tqdm(val_loader):
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                val_loss += loss.item() * noisy.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        # show_denoising_result(model,val_loader)

# 모델 저장
torch.save(model.state_dict(), 'dncnn.pth')
