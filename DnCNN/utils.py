import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch
import numpy as np

def show_denoising_result(model,dataloader,idx=0,device="cpu"):
    model.eval()
    noisy_batch, clean_batch = next(iter(dataloader))

    # 하나의 샘플만 사용
    noisy = noisy_batch[idx:idx+1].to(device)      # [1, C, H, W]
    clean = clean_batch[idx].permute(1, 2, 0).numpy()  # [H, W, C]

    with torch.no_grad():
        output = model(noisy).squeeze(0).cpu().permute(1, 2, 0).numpy()

    noisy = noisy.squeeze(0).cpu().permute(1, 2, 0).numpy()

    # [0,1] → [0,255]
    noisy_img = (noisy * 255).astype(np.uint8)
    clean_img = (clean * 255).astype(np.uint8)
    output_img = np.clip(output * 255, 0, 255).astype(np.uint8)

    # 시각화
    plt.figure(figsize=(12, 4))
    titles = ['Noisy Input', 'DnCNN Output', 'Clean Target']
    images = [noisy_img, output_img, clean_img]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()