# Project Title: Audio Classification via Image Recognition
# Methodology: Convolutional Neural Network for Image Classification on Mel-Scale Spectrograms
# Author: Griffin Dean Kent
import torch
import torch.nn as nn
import numpy as np
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torchaudio
from PIL import Image
import torchvision.transforms as transforms
import torchaudio.transforms as T
import torchvision.models as models



##############################################################################################################################################################
#################################################################### Spectrogram Functions ###################################################################
##############################################################################################################################################################
#Standard function to convert an audio file to a spectrogram
def audio_to_spect(path, n_fft=1024, hop_length=512, n_mels=200):
    """
    A Standard function that converts a .wav audio file into mel-scale spectrograms
    
    Parameters
    ----------
    path : The path to the audio file
    n_fft : Window size for the short-time Fourier Transform
    hop_length : The step-size between successive frames
    n_mels : The number of mel filterbanks
    """
    wave, sample_rate = torchaudio.load(path)
    spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)(wave)
    spectrogram = np.log(spectrogram + 1e-10)  # Log transform for better visibility (the "+ 1e-10" prevents from taking log of 0)
    return spectrogram

#Function to turn the spectrograms into images (this will be used to load the data as images instead of performing the Fourier transform every time we access an audio)
def audio_to_spect_image(path, out_path):
    spect = audio_to_spect(path)[0]
    plt.figure(figsize=(5, 5))
    plt.pcolormesh(spect.numpy(), cmap='magma')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

#Standard function to visualize a spectrogram
def vis_spect(spect):
    plt.figure(figsize=(10, 5))
    plt.imshow(spect.log2()[0, :, :].numpy(), cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()
    
def vis_spect_image(image, label):
    image = image.permute(1, 2, 0).numpy()  # Change shape to (224, 224, 3)
    image = (image - image.min()) / (image.max() - image.min())  # Normalize for display
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='magma')
    plt.colorbar()
    plt.title(f"Spectrogram - Class {label}")
    plt.show()
    

##############################################################################################################################################################
##################################################################### Define the Dataset #####################################################################
##############################################################################################################################################################
from torch.utils.data import Dataset, DataLoader

class Audio_Dataset(Dataset):
    """
    This is a general class to define a dataset for audio samples
    """
    def __init__(self, path, transform = None):
        self.path = path
        self.transform = transform
        self.audio_files = []
        for genre in os.listdir(path):
            genre_path = os.path.join(path, genre)
            for audio_file in os.listdir(genre_path):
                self.audio_files.append((os.path.join(genre_path, audio_file), genre))
        self.genre_labels = {genre: index for index, genre in enumerate(os.listdir(path))}
    
    def __len__(self):
        #Returns the total number of audio files
        return len(self.audio_files)
    
    def __getitem__(self, index):
        #Gets the path of the audio file at the given index
        file_path, genre = self.audio_files[index]
        #Turn the audio .wav file into a spectrogram
        spect = audio_to_spect(file_path)
        #Apply transforms if there are any
        if self.transform:
            spect = self.transform(spect)
        #Get the genre label
        label = self.genre_labels[genre]
        #Return the spectrogram and corresponding audio file
        return spect, label

class Spect_Image_Dataset(Dataset):
    """
    This is a general class to define a dataset for audio samples where where the files have been converted to images of spectrograms
    """
    def __init__(self, path, transform = None):
        self.path = path
        self.transform = transform
        self.spect_files = []
        for genre in os.listdir(path):
            genre_path = os.path.join(path, genre)
            for spect_file in os.listdir(genre_path):
                self.spect_files.append((os.path.join(genre_path, spect_file), genre))
        self.labels = {genre: index for index, genre in enumerate(os.listdir(path))}
        self.genre_to_index = {idx: genre for genre, idx in self.labels.items()}
    
    def __len__(self):
        #Returns the total number of audio files
        return len(self.spect_files)
    
    def __getitem__(self, index):
        #Gets the path of the audio file at the given index
        file_path, genre = self.spect_files[index]
        image = Image.open(file_path).convert("RGB")
        #Apply transforms if there are any
        if self.transform != None:
            image = self.transform(image)
        #Get the genre label
        label = self.labels[genre]
        #Return the spectrogram and corresponding audio file
        return image, label

#Data augmentation transforms
def random_noise(image):
    noise = torch.randn_like(image) * 0.1 #Generates Gaussian random noise in the dimensions of the image
    return image + noise

class UseWithProb:
    def __init__(self, transform, prob = 0.5):
        self.transform = transform
        self.prob = prob
        
    def __call__(self, x):
        if random.random() < self.prob:
            return self.transform(x)
        else:
            return x

#Define the standard transformations to apply to the images
transformations = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

#Define the augmentations to apply to the images
augmentation_transformations = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop(200),
        UseWithProb(transforms.RandomResizedCrop(200, scale=(0.8, 1.0)), prob=0.6),
        #UseWithProb(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)), prob=0.5),
        transforms.ToTensor(),
        UseWithProb(transforms.Lambda(lambda x: random_noise(x)), prob=0.6),
        UseWithProb(T.TimeMasking(time_mask_param=50), prob=0.6),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#Set seed function
def Set_Seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

##############################################################################################################################################################
###################################################################### Training Algorithm ####################################################################
##############################################################################################################################################################
def train_algo(model, dataloaders, optimizer, loss_func, device, num_epochs, lr_scheduler, sched_type):
    """
    Training algorithm
    """
    #Set up an interactive plot
    plt.ion()
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    line, = ax.plot(xdata, ydata, '-')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training ResNet-18 for Audio Classification via Transfer Learning', size=12)
    ax.grid(True)
    
    train_dataloader = dataloaders[0]
    val_dataloader = dataloaders[1]
    iter_list = []
    loss_list = []
    val_loss_list = []
    i = 0
    for epoch in range(num_epochs):
        epoch_error_list = []
        for (x,y) in train_dataloader:
            model.train()
            iter_list.append(i)
            i+=1
            
            x, y = x.to(device), y.to(device)
            #Model prediction
            preds = model(x)
            loss = loss_func(preds, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            epoch_error_list.append(loss.item())
            
            if sched_type == "one_cycle":
                lr_scheduler.step()
            
            #Compute the validation loss
            model.eval()
            with torch.no_grad():
                x_val, y_val = next(iter(val_dataloader))
                x_val, y_val = x_val.to(device), y_val.to(device)
                #Model prediction
                val_preds = model(x_val)
                val_loss = loss_func(val_preds, y_val)
                val_loss_list.append(val_loss.item())
            
            #Live plotting
            xdata.append(i)
            ydata.append(loss.item())
            line.set_xdata(xdata)
            line.set_ydata(ydata)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        avg_loss = np.sum(np.asarray(epoch_error_list)) / len(epoch_error_list)
        if sched_type == "plateau":
            lr_scheduler.step(avg_loss)
        #print(optimizer.param_groups[0]['lr'])
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")
        plt.ioff() #Turn the interactive plot off
    return iter_list, loss_list, val_loss_list


##############################################################################################################################################################
##################################################################### Evaluation Functions ###################################################################
##############################################################################################################################################################
# Function to plot a batch of spectrograms with true and predicted labels
def plot_batch_predictions(model, data_loader, classes, device):
    model.eval()
    images, labels = next(iter(data_loader))  # Get a batch
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    batch_size = images.shape[0]
    grid_size = int(batch_size ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    correct = 0
    
    for i, ax in enumerate(axes.flat):
        if i >= batch_size:
            break
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
        ax.imshow(img)
        ax.set_title(f"True Label: {classes[labels[i].item()]}\nPred Label: {classes[preds[i].item()]}")
        ax.axis("off")
        if labels[i] == preds[i]:
            correct += 1
    
    accuracy = (correct / batch_size) * 100
    fig.suptitle(f"Audio-Net Batch Accuracy: {accuracy:.2f}%", fontsize=16)
    plt.show()


##############################################################################################################################################################
################################################ Turn Audio Files Into Images of Spectrograms (Only Run Once) ################################################
##############################################################################################################################################################
orig_path = "INSERT PATH HERE"
spect_dir_path = "INSERT PATH HERE"
for genre in os.listdir(orig_path):
    genre_path = os.path.join(orig_path, genre)
    out_path = os.path.join(spect_dir_path, genre)
    os.makedirs(out_path, exist_ok=True) #It is the "exist_ok=True" that makes this safe to run multiple times, even when the director already exists
    for file in os.listdir(genre_path):
        if file.endswith('.wav'):
            audio_path = os.path.join(genre_path, file)
            out_file = os.path.join(out_path, file.replace(".wav",".png"))
            audio_to_spect_image(audio_path, out_file)
    

##############################################################################################################################################################
############################################################################ Main ############################################################################
##############################################################################################################################################################
def main():
    ###########################################################################
    ######################### Load the Data (Audio) ###########################
    ###########################################################################
    orig_path = "INSERT PATH HERE"
    batch_size = 32
    #Setup the dataset and dataloader
    gtzan_dataset = Audio_Dataset(orig_path)
    data_loader = DataLoader(gtzan_dataset, batch_size=batch_size)
    #Visualize some of the spectrograms
    for batch in data_loader:
        spectrograms, filenames = batch
        print(filenames)
        print(spectrograms.shape)
        vis_spect(spectrograms[0])
        break
    
    ###########################################################################
    ###################### Load the Data (Spectrograms) #######################
    ###########################################################################
    spect_images_path = "INSERT PATH HERE"
    train_batch_size = 100
    val_batch_size = 80
    Set_Seed(42)
    #Setup the dataset and dataloader
    gtzan_dataset = Spect_Image_Dataset(spect_images_path, augmentation_transformations)
    #Set up train-val splits of the data
    train_size = int(0.8*len(gtzan_dataset))
    val_size = len(gtzan_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(gtzan_dataset, [train_size, val_size])
    #Set up the data loaders
    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=True)
    dataloaders = [train_loader, val_loader]
    #Visualize some of the spectrograms
    for batch in train_loader:
        spectrograms, labels = batch
        print(labels)
        print(spectrograms.shape)
        vis_spect_image(spectrograms[0], labels[0])
        break
    
    ###########################################################################
    ########################### Model and Training ############################
    ###########################################################################
    #Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 3e-4
    weight_decay = 0.9
    num_epochs = 100
    
    #Load a pre-trained model to perform transfer learning on
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features #The number of nodes in the second-to-last layer
    model.fc = nn.Linear(num_features, len(gtzan_dataset.labels)) #Define the number of output nodes to match the number of classes for our data
    
    #Determine the number of layers to freeze to perform transfer learning
    for weights in list(model.parameters())[:-5]:
        weights.requires_grad = False
    
    #Define loss, optimizer, and scheduler
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, epochs=num_epochs, steps_per_epoch=len(train_loader))
    sched_type = "one_cycle" #"none" #"one_cycle" #"plateau"
    
    #Train the model
    model.to(device)
    iter_list, train_loss_list, val_loss_list = train_algo(model, dataloaders, optimizer, loss_func, device, num_epochs, lr_scheduler, sched_type)
    #Plotting
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    sns.lineplot(x=iter_list, y=train_loss_list, linewidth=0.3, label='Training Loss', legend=True, color='blue', linestyle='-').set_title('Audio-Net: Trained ResNet-18 for Audio Classification', size=15)
    sns.lineplot(x=iter_list, y=val_loss_list, linewidth=0.3, label='Validation Loss', legend=True, color='red', linestyle='-')
    plt.xlabel("Iterations")
    plt.ylabel("Cross-Entropy Loss")
    axes.grid(True)
    plt.show()
    
    #Evaluate the accuracy of the model on a batch (with no augmentations applied to them) and display the results in a grid
    clean_dataset = Spect_Image_Dataset(spect_images_path, transformations)
    size=9
    clean_loader = DataLoader(clean_dataset, size, shuffle=True)
    plot_batch_predictions(model, clean_loader, clean_dataset.genre_to_index, device)
    
    


if __name__ == "__main__":
    main()


