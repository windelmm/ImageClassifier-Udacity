# Import libraries
import argparse
import datetime

import torch
import torchvision

from torch import nn, optim
from torchvision import datasets,transforms, models
from collections import OrderedDict

def data_preparation(data_container, training_batch, validation_testing_batch):
    data_dir = data_container
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Transforms for the training, validation, and testing sets
    transforms_training = transforms.Compose([
                                            transforms.RandomHorizontalFlip(p=0.25),
                                            transforms.RandomRotation(25),
                                            transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))
                                            ])
                                            
    #No rotation or scaling, only resize
    transforms_validation = transforms.Compose([
                                                transforms.Resize(225),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                    (0.229, 0.224, 0.225))
                                            ])

    #No rotation or scaling, only resize
    transforms_testing = transforms.Compose([
                                            transforms.Resize(225),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))
                                            ])


    # Load the three datasets (with Imagefolder) and use the above transformations
    dataset_training = datasets.ImageFolder(train_dir, transform = transforms_training)
    dataset_validation = datasets.ImageFolder(valid_dir, transform = transforms_validation)
    dataset_testing = datasets.ImageFolder(test_dir, transform = transforms_testing)

    # Define the dataloaders
    dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=training_batch, shuffle=True, drop_last=True)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=validation_testing_batch)
    dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=validation_testing_batch)

    return dataloader_training, dataloader_validation, dataloader_testing, dataset_training

def pretrained_model(model_name):
    model = None
    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    if model_name == "vgg19":
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    return model


def define_classifier(input_size,hidden_sizes):
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, 512)),
            ('relu1', nn.ReLU()),
            ('drp1', nn.Dropout(p=0.5)),
            ('hidden', nn.Linear(512, hidden_sizes[0])),
            ('fc2', nn.Linear(hidden_sizes[1], 102)),
            ('output', nn.LogSoftmax(dim=1))]))
    return classifier


def training(epochs, training_batch, validation_testing_batch, gpu):
    
    #Set gradients to zero. 
    model.zero_grad()
    
    #Set criterion to measure.
    criterion = nn.CrossEntropyLoss()
    
    for e in range(epochs):
        start = datetime.datetime.now()
        
        for e in range(epochs):
            total = 0
            correct = 0
            print(f'Epoch {e+1}\n')
            for i, (images, labels) in enumerate(dataloader_training):
                
                if gpu:
                    images = images.to('cuda')
                    labels = labels.to('cuda')
                
                # Set gradients of all parameters to zero. 
                optimizer.zero_grad()
                
                # Propigate forward and backward 
                outputs = model.forward(images)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            #Epoch result
            e_valid_correct = 0
            e_valid_total = 0
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(dataloader_validation):
                    images = images.to('cuda')
                    labels = labels.to('cuda')
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    e_valid_total += labels.size(0)
                    e_valid_correct += (predicted == labels).sum().item()
                correct_perc = 0
                if e_valid_correct > 0:
                    correct_perc = (100 * e_valid_correct // e_valid_total)
                print(f'Accuracy {correct_perc:d}%\n')

    end = datetime.datetime.now()
    total_duration = end - start
    print('Time epoch: {} mins'.format(total_duration.total_seconds()/60))

def validation(model, dataloader_validation, criterion, validation_testing_batch, gpu):
    # DONE: Do validation on the test set
    correct = 0
    total = 0
    
    # Disabling gradient calculation
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader_validation):
            images = images.to('cuda')
            labels = labels.to('cuda')
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy  {(100 * correct // total):d}%\n ')

def testing(model, dataloader_testing, criterion, validation_testing_batch, gpu):
    # DONE: Do validation on the test set
    correct = 0
    total = 0
    
    # Disabling gradient calculation
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader_testing):
            images = images.to('cuda')
            labels = labels.to('cuda')
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy  {(100 * correct // total):d}%\n ')


def save_checkpoint(model, model_name, criterion, optimizer, dataset_training, epochs, saving_path):
    checkpoint = {
            'model_state': model.state_dict(),
            'criterion_state': criterion.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'class_to_idx': dataset_training.class_to_idx,
            'classifier': classifier,
            'epoch': epochs,
            'model_name': model_name
           }

    torch.save(checkpoint, saving_path + 'trained_model.pth')
    print("Saved to " + saving_path + "trained_model.pth")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Deep learning model")
    parser.add_argument("-g", "--gpu", help="use GPU", default=False, action="store_true")     
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of epochs")     
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate")    
    parser.add_argument("-a", "--arch", type=str, default="vgg19", help="pretrained model") 
    parser.add_argument("-d", "--data_container", type=str, required=True, help="data_container")        
    parser.add_argument("-u", "--hidden_units", type=int, nargs='+', required=True, help="number of hidden units")
    parser.add_argument("-s", "--save_dir", type=str, default="./", help="save_directory")
    
    args = parser.parse_args()

    print("GPU: " + str(args.gpu))
    print("epochs: " + str(args.epochs))
    print("Learning Rate: " + str(args.learning_rate))
    print("Pretrained model: " + args.arch)
    print("Hidden units: " + str(args.hidden_units))
    print("Saving model to: " + str(args.save_dir))

    training_batch = 64
    validation_testing_batch = 32
    input_size = 25088
    output_size = 102
    
    dataloader_training, dataloader_validation, dataloader_testing, dataset_training = data_preparation(args.data_container, training_batch, validation_testing_batch)

    model = pretrained_model(args.arch)

    classifier = define_classifier(input_size,args.hidden_units)

    model.classifier = classifier

    # Move model to GPU
    if args.gpu:
        model.to('cuda')

    # Optimizer should be defined after moving the model to GPU
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train
    training(args.epochs, training_batch, validation_testing_batch, args.gpu)

    # Test
    testing(model, dataloader_testing, criterion, validation_testing_batch, args.gpu)

    # Save model
    save_checkpoint(model, args.arch, criterion, optimizer, dataset_training, args.epochs, args.save_dir)