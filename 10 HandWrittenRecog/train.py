import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from cnn import CNN,train,test
plt.rcParams['font.sans-serif']    = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
plt.rcParams['figure.dpi']         = 100        #分辨率

data_path = "../pytorch practice/10 HandWrittenRecog/dataset"
save_dir = "../pytorch practice/10 HandWrittenRecog/savepoints"
save_file = "checkpoints.pt"
save_path = os.path.join(save_dir,save_file)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])

train_dataset = datasets.MNIST(root=data_path,train=True,download=True,transform=transform)
tests_dataset = datasets.MNIST(root=data_path,train=False,download=True,transform=transform)

batch_size = 32
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
tests_loader = DataLoader(dataset=tests_dataset,batch_size=batch_size,shuffle=True)

check_points = None
if os.path.exists(save_path):
    check_points = torch.load(save_path)

model = CNN()
model.to(device)
loss_fc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)
if(check_points): 
    model.load_state_dict(check_points["model"])
    optimizer.load_state_dict(check_points["optimizer"])

train_loss_list = check_points["train_loss"] if check_points else []
train_acc_list = check_points["train_acc"] if check_points else []
test_loss_list = check_points["test_loss"] if check_points else []
test_acc_list = check_points["test_acc"] if check_points else []

epoches = 10
for epoch in range(epoches):
    model.train()
    train_loss,train_acc = train(model=model,loss_fc=loss_fc,optimizer=optimizer,dataloader=train_loader,device=device)
    model.eval()
    test_loss,test_acc = test(model=model,loss_fc=loss_fc,dataloader=tests_loader,device=device)

    print(f"第{epoch+1}轮，训练损失为{train_loss:.4f}，准确率为{train_acc*100:.2f}%，测试损失为{test_loss:.4f}，测试准确率为{test_acc*100:.2f}%")
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

plt.figure(figsize=(12,6))
xrange = [i+1 for i in range(len(test_acc_list))]

plt.subplot(1,2,1)
plt.plot(xrange,train_loss_list,label="train_loss")
plt.plot(xrange,test_loss_list,label="test_loss")
plt.title("train & test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(xrange,train_acc_list,label="train_acc")
plt.plot(xrange,test_acc_list,label="test_acc")
plt.title("train & test accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()

plt.show()

save_dic = {
    "model":model.state_dict(),
    "optimizer":optimizer.state_dict(),
    "train_loss":train_loss_list,
    "train_acc":train_acc_list,
    "test_loss":test_loss_list,
    "test_acc":test_acc_list
}
torch.save(save_dic,save_path)