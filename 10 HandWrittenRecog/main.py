import torch
import torch.nn as nn
import imgcv
import cnn
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2
plt.rcParams['font.sans-serif']    = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
plt.rcParams['figure.dpi']         = 100        #分辨率

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "../pytorch practice/10 HandWrittenRecog/dataset"
save_dir = "../pytorch practice/10 HandWrittenRecog/savepoints"
save_file = "checkpoints.pt"
save_path = os.path.join(save_dir,save_file)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

"""图像读取和处理"""
img_dir = "../pytorch practice/10 HandWrittenRecog/image"
image_paths = []
image_files = []
for imgfile in os.listdir(img_dir):
    image_files.append(imgfile)
    image_path = os.path.join(img_dir,imgfile)
    image_paths.append(image_path)

imgs = []
for image_path in image_paths:
    imgs.append(imgcv.imageSplit(image_path,True))

"""读取模型并开始预测"""
checkpoints = None
if os.path.exists(save_path):
    checkpoints = torch.load(save_path)

model = cnn.CNN()
model.to(device)
loss_fc = nn.CrossEntropyLoss()
if checkpoints:
    model.load_state_dict(checkpoints["model"])
model.eval()
with torch.no_grad():
    for i in range(len(imgs)):
        digit_imgs = imgs[i]
        print(f"第{i+1}张图片：{image_files[i]}")
        image = cv2.imread(image_paths[i],cv2.IMREAD_GRAYSCALE)
        recogDigits = ""
        for digit_img in digit_imgs:
            """转化为图片，并且变成灰度、28x28且标准化的张量"""
            img = Image.fromarray(digit_img)
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28,28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
            ])
            img = transform(img)
            img = img.to(device)
            img = img.unsqueeze(dim=1)  #增加batch维度

            pred = model(img)
            digit = pred.argmax(dim=1).item()
            recogDigits += str(digit)
        plt.figure()
        plt.imshow(image)
        plt.title(recogDigits)
        plt.show()

