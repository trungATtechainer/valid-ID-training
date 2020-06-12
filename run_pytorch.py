import torch
from torchvision import datasets, transforms, models
import numpy as np
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1033120000
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import helper
from tqdm import tqdm

from tensorboardX import SummaryWriter
from trains import Task
import warnings
warnings.filterwarnings("ignore")

task = Task.init(project_name = 'Valid ID Classify',
                task_name = 'Stage 2 - MobileNet - Pytorch')

writer = SummaryWriter()

# define transformation for images that are put into the dataset on here
transform = transforms.Compose([transforms.Resize(225),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# importing our data using torchvision.datasets.
train_set = datasets.ImageFolder("data/Train", transform = transform)
test_set = datasets.ImageFolder("data/Valid", transform = transform)

# put data into a Dataloader using torch
train_loader = torch.utils.data.DataLoader(train_set, batch_size= 128,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= 128, shuffle = True)

import torch.nn as nn
import torch.optim as optim
# Using a pretrained model
model = models.mobilenet_v2(pretrained = True)

# turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Dropout(0.2, inplace = False),
              nn.Linear(1280, 512, bias = True),
              nn.Dropout(0.2, inplace = True),
              nn.Linear(512, 3, bias = True))

model.classifier = classifier

# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to the device specified above
model.to(device)

# Set the error function using torch.nn as nn library
criterion = nn.CrossEntropyLoss()

# Set the optimizer function using torch.optim as optim library
optimizer = optim.SGD(model.classifier.parameters(), lr = 0.01, momentum=0.9)

global model_name

model_name = 0

def train(epochs):
    for epoch in range(epochs):
        train_loss = 0
        test_loss = 0
        accuracy = 0
        
        # Training the model
        model.train()
        for inputs, labels in tqdm(train_loader):
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear optimizers
            optimizer.zero_grad()
            # Forward pass
            output = model.forward(inputs)
            # Loss
            loss = criterion(output, labels)
            # Calculate gradients (backpropogation)
            loss.backward()
            # Adjust parameters based on gradients
            optimizer.step()
            # Add the loss to the training set's rnning loss
            train_loss += loss.item()*inputs.size(0)
        
        global model_name
        torch.save(model, 'pytorch_model/' + model_name + '_pytorch_v1.pth')

        model_name += 1

        # Evaluating the model
        model.eval()
        # Tell torch not to calculate gradients
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                output = model.forward(inputs)
                # Calculate Loss
                testloss = criterion(output, labels)
                # Add loss to the validation set's running loss
                test_loss += testloss.item()*inputs.size(0)
                
                # Since our model outputs a LogSoftmax, find the real 
                # percentages by reversing the log function
                output = torch.exp(output)
                # Get the top class of the output
                top_p, top_class = output.topk(1, dim=1)
                # See how many of the classes were correct?
                equals = top_class == labels.view(*top_class.shape)
                # Calculate the mean (get the accuracy for this batch)
                # and add it to the running accuracy for this epoch
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()            
                
        
        # Get the average loss for the entire epoch
        train_loss = train_loss/len(train_loader.dataset)
        test_loss = test_loss/len(test_loader.dataset)
        
        writer.add_scalar('data/train loss', train_loss, epoch)
        writer.add_scalar('data/valid loss', test_loss, epoch)
        writer.add_scalar('data/accuracy', accuracy/len(test_loader), epoch)
        # Print out the information
        print('Accuracy: ', accuracy/len(test_loader))
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, test_loss))

if __name__ == "__main__":
    train(2)
    for param in model.parameters():
        param.requires_grad = True
    train(10)