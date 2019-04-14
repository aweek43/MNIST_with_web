from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

################################################

input_img = Image.open('input/input.jpg').convert('L')
input_img = input_img.resize((28, 28))
img_array = numpy.array(input_img)

for i in range(28):
    for j in range(28):
            img_array[i][j] = 255 - img_array[i][j]

mnist_data = Image.fromarray(img_array)
plt.imshow(mnist_data)
mnist_data.save('mnist_data/mnist_data.jpg')

# data_loader

def mnist_image_loader(path):
    return Image.open(path)

data_dir = 'mnist_data'          # this path depends on your computer
input_data = datasets.ImageFolder(data_dir, transforms.ToTensor(), loader=mnist_image_loader)
input_loader = torch.utils.data.DataLoader(dataset=input_data)

#################################################

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()
model.load_state_dict(torch.load('mnist_model/mnist_model.pth'))
# model.eval() # for training

result = -1
pred = -1

for data in list(input_loader)[0]:
    data = torch.FloatTensor(data)
    data = Variable(data, volatile=True)
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    break

result = pred[0][0].numpy()
print("주어진 사진의 숫자는 ", result, "입니다.")

