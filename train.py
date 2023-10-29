from data_process.data_loader import data_loader
import torch.optim as optim
from model.net import Net
import torch.nn as nn
import torch


# init
total_epoch = 1000
device = torch.device("cuda:0")
net = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train
dir = "./data/"
imgs, gts = data_loader(dir)
running_loss = 0.0
for ep_i in range(total_epoch):
    for i, (img, gt) in enumerate(zip(imgs, gts)):
        optimizer.zero_grad()
        input = torch.from_numpy(img).float().to(device).permute(2, 0, 1).unsqueeze(0)
        preds = torch.sigmoid(net(input))
        gt = torch.from_numpy(gt).float().to(device).flatten()
        loss = criterion(preds, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {ep_i} Loss: {running_loss}")
    running_loss = 0.0

# save checkpoint
ckp_name = "./checkpoint/sfd_v0.pth"
torch.save(net.state_dict(), ckp_name)
print(f"{ckp_name} is saved ")
