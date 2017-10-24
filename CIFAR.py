import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

Li = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#Taking Training Class Input from User
classes_Train=[]
print("Please select any of below classes you want to train network on: \n 'plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'")
for i in range(0,2):
    classes_Train.insert(i,input("Enter the Train Image Class Name in lowercase: "))

#Finding the specific Class Labels for the Train Classes Entered
Train_classes_Labels=[]
for index,i in enumerate(classes_Train):
    Train_classes_Labels.insert(index,Li.index(i))

#Taking Testing Class Inputs from User
classes_Test=[]
print("\nPlease select any of below classes you want to test network on: \n 'plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'")
for i in range(0,2):
    classes_Test.insert(i,input("Enter the Images Class Name in lowercase: "))

#Finding the specific Class Labels for the Test Classes Entered
Test_classes_Labels=[]
for index,i in enumerate((classes_Test)):
    Test_classes_Labels.insert(index,Li.index(i))

print(classes_Train)
print(Train_classes_Labels)
print(classes_Test)
print(Test_classes_Labels)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=2)

classes = tuple(classes_Train)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
#print(labels)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
#print(labels[0])
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        #self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = self.fc4(x)
        return x
def Test_Images():
    nooftst=0
    nooftst1=0
    correct = 0
    total = 0
    for data in testloader:
        nooftst=nooftst+1
        images, labels = data
        x=labels.numpy()
        classcount=list(x.flatten()).count(Test_classes_Labels[0]) + list(x.flatten()).count(Test_classes_Labels[1])    
        if classcount>=1:
            if list(x.flatten()).count(Test_classes_Labels[0])==1:
                labels=0
            else:
                labels=1
            labels=(torch.LongTensor([labels]))         
            nooftst1=nooftst1+1
            outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print('Accuracy of the network on the 40000 test images: %d %%' % (100 * correct / total))
    print(correct)

net = Net()
net1 = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        x=labels.data.numpy()
        classcount=list(x.flatten()).count(Train_classes_Labels[0]) + list(x.flatten()).count(Train_classes_Labels[1])
        # wrap them in Variable
        if classcount>=1:
            # zero the parameter gradients
            #print("came here")
            #print(type(labels))
            if list(x.flatten()).count(Train_classes_Labels[0])==1:
                labels=0
            else:
                labels=1
            #print(outputs)
            #print(labels)
            labels=Variable(torch.LongTensor([labels]))
            #labels=str(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            # forward + backward + optimize
            #net.conv1.weight[0]=w1
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    Test_Images()        #print(labels)
        #else:
            #print("less no of labels to do train in the batch")

print('Finished Training ')
