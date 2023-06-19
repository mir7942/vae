import os
import random
import numpy as np
import torch
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from models import *

RANDOM_SEED = 2
DATASET_PATH = './data/'
IMAGE_PATH = './images/'
BATCH_SIZE = 128
NUM_WORKERS = 1
INPUT_DIM = 28 * 28
HIDDEN_DIM = 400
LATENT_DIM = 10
DROPOUT = 0.2
LEARNING_RATE = 1e-3
N_EPOCHS = 30

def loss_function(x, x_reconst, mean, log_var):
    reconst_loss = nn.functional.binary_cross_entropy(x_reconst, x, reduction='sum')
    kl_div = 0.5 * torch.sum(mean.pow(2) + log_var.exp() - log_var - 1)
    
    return reconst_loss + kl_div, reconst_loss, kl_div

def train():
    print("PyTorch version: " + torch.__version__)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    os.makedirs(IMAGE_PATH, exist_ok=True) 

    # CPU 또는 GPU 선택
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Dataset
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
    testset = datasets.MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    encoder = Encoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, dropout=DROPOUT)
    decoder = Decoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM, dropout=DROPOUT)

    model = Model(Encoder=encoder, Decoder=decoder).to(device)

    print(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)    

    for epoch in range(1, N_EPOCHS+1):
        train_loss = 0
    
        for i, (x, _) in enumerate(train_dataloader):
            # Forward
            x = x.view(-1, INPUT_DIM)
            x = x.to(device)

            optimizer.zero_grad()

            x_reconst, mean, log_var = model(x)

            # Compute reconstruction loss and KL divergence
            loss, reconst_loss, kl_div = loss_function(x, x_reconst, mean, log_var)

            # backprop and optimize            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 10 == 0:
                print(f'Epoch [{epoch}/{N_EPOCHS}], Step [{i}/{len(train_dataloader)}], Reconstruction Loss : {reconst_loss.item():.4f}, KL Divergence: {kl_div.item():.4f}')

        print(f'===> Epoch: {epoch} Average Train Loss: {train_loss/len(train_dataloader.dataset):.4f} ')
        
        test_loss = 0
        with torch.no_grad():
            for i, (x, _) in enumerate(test_dataloader):
                # Forward
                x = x.view(-1, INPUT_DIM)
                x = x.to(device)

                x_reconst, mean, log_var = model(x)

                # Compute reconstruction loss and KL divergence
                loss, _, _ = loss_function(x, x_reconst, mean, log_var)

                test_loss += loss.item()

                # save reconstruction images
                if i==0:
                    x_concat = torch.cat([x.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
                    # batch size 개수만큼의 이미지 쌍(input x, reconstructed x)이 저장됨
                    save_image(x_concat, os.path.join(IMAGE_PATH,f'reconst-epoch{epoch}.png'))

            print(f'===> Epoch: {epoch} Average Test Loss: {test_loss/len(test_dataloader.dataset):.4f} ')
            
            # save sampled images
            z = torch.randn(BATCH_SIZE, LATENT_DIM).to(device) # N(0, 1)에서 z 샘플링
            sampled_images = decoder(z)
            save_image(sampled_images.view(-1, 1, 28, 28), os.path.join(IMAGE_PATH,f'sampled-epoch{epoch}.png'))

if __name__ == '__main__':
    train()