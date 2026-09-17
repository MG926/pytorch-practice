import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1600,50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50,10)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        return x
    
def train(model,loss_fc,dataloader,optimizer,device):
    average_acc = 0
    average_loss = 0
    totol_num = 0
    for batch_x,batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = loss_fc(pred,batch_y)

        loss.backward()
        optimizer.step()

        batch_s = batch_y.size(0)
        totol_num += batch_s
        average_loss += loss.item()*batch_s
        average_acc += (pred.argmax(dim=1)==batch_y).float().sum().item()
    average_acc /= totol_num
    average_loss /= totol_num
    return average_loss,average_acc

def test(model,loss_fc,dataloader,device):
    average_acc = 0
    average_loss = 0
    totol_num = 0
    with torch.no_grad():
        for batch_x,batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = loss_fc(pred,batch_y)

            batch_s = batch_y.size(0)
            totol_num += batch_s
            average_loss += loss.item()*batch_s
            average_acc += (pred.argmax(dim=1)==batch_y).float().sum().item()
        average_acc /= totol_num
        average_loss /= totol_num
    return average_loss,average_acc