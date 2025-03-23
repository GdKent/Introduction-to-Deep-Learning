# Project Title: Multi-Net
# Methodology: Multi-Modal Architecture for Video Recommendation
# Author: Griffin Dean Kent
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torchaudio
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
import torchvision.models as models
import json
import av




##############################################################################################################################################################
##################################################################### Multi-Modal Network ####################################################################
##############################################################################################################################################################

###############################################################################
############################## Modality Encoders ##############################
###############################################################################

#Video Encoder
def import_ResNet():
    """
    This function builds a pretrained ResNet model (either 18 or 50) for generating the video frame embeddings.
    This model has ~11.7M parameters.
    """
    resnet_model = models.resnet18(pretrained=True)
    #Remove the final fully-connected layer to return the final embedded layer
    out_dim = resnet_model.fc.in_features
    resnet_model.fc = nn.Identity()
    return resnet_model, out_dim

#Audio Encoder
def import_HuBERT(model_path="facebook/hubert-base-ls960"):
    """
    This function builds a pretrained HuBERT (Hidden-Unit BERT) model from Hugging Face's Transformers.
    This is an encoder transformer for speech representation learning. This will be used to generate
    embeddings for the audio data. This model has ~96M parameters.
    """
    hubert_model = AutoModel.from_pretrained(model_path)
    out_dim = hubert_model.config.hidden_size
    return hubert_model, out_dim

#Text Encoder
def import_ALBERT(model_path="albert/albert-base-v2"):
    """
    This function builds a pretrained ALBERT model from Hugging Face's Transformers along.
    This is an encoder transformer for text representation learning. This will be used to generate
    embeddings for the text data. The reason we use ALBERT instead of BERT is because ALBERT has ~12M parameters
    whereas BERT has ~110M parameters.
    """
    albert_model = AutoModel.from_pretrained(model_path)
    out_dim = albert_model.config.hidden_size
    return albert_model, out_dim


###############################################################################
###################### Multi-Modal Backbone Architecture ######################
###############################################################################
class MultiModalModel(nn.Module):
    """
    This is the multi-modal network that combines all of the different modality encoders. It consists of the following:
        - A ResNet model to generate embeddings for the 2D sampled frames from the videos.
        - A HuBERT model to generate embeddings for the audio data.
        - An ALBERT model to generate embeddings for the text data.
    Further, this model combines the three modality embeddings in an early fusion design, followed by some number of
    fully-connected layers in an MLP.
    
    Parameters
    ----------
    vid_embd_dim : The dimension of the output vector of the ResNet model
    aud_embd_dim : The dimension of the output vector of the HuBERT model
    tex_embd_dim : The dimension of the output vector of the ALBERT model
    aggregated_dim : The dimension of the final embedding size for each of the modalities
    
    """
    def __init__(self, aggregated_dim, device):
        super().__init__()
        self.video_encoder, vid_embd_dim = import_ResNet() #Build the video encoder
        self.audio_encoder, aud_embd_dim = import_HuBERT() #Build the audio encoder
        self.text_encoder, tex_embd_dim = import_ALBERT() #Build the text encoder
        #self.video_encoder.to(device); self.audio_encoder.to(device); self.text_encoder.to(device)
        #Create the layers that project each of the embeddings to a smaller and unified dimension size
        self.video_proj = nn.Linear(vid_embd_dim, aggregated_dim)#.to(device)
        self.audio_proj = nn.Linear(aud_embd_dim, aggregated_dim)#.to(device)
        self.text_proj = nn.Linear(tex_embd_dim, aggregated_dim)#.to(device)
        self.device = device
        #Create the final fully-connected MLP
        self.mlp = nn.Sequential(nn.Linear(aggregated_dim * 3, 700),
                                 nn.ReLU(),
                                 nn.Linear(700, 500),
                                 nn.ReLU(),
                                 nn.Linear(500, aggregated_dim))#.to(device)
    
    def forward(self, video_frames, batch_audio, batch_text):
        """
        Parameters
        ----------
        video_frames : A batch of sampled 2D frames for each video in the batch [batch_size, N, C, H, W], where N is the number of sampled frames per video,
                       C is the number of channels, H is the height, and W is the width.
        audio : 
        text : A list of strings (captions) that have been attributed to each video.
        
        Returns
        -------
        A tensor of shape [batch_size, aggregated_dim] which represents the fused multi-model embedding.
        """
        #-------------------------------
        #------- Video Embedding -------
        #-------------------------------
        batch_size, N, C, H, W = video_frames.shape
        #We need to reshape the video data for resnet
        video_frames = video_frames.view(batch_size * N, C, H, W).to(self.device)
        #Pass the video data through the ResNet model
        frame_embeddings = self.video_encoder(video_frames) #This will result in a shape [batch_size * N, vid_embd_dim]
        #Reshape the embeddings back and average them over the samples per video (this will average them over time)
        frame_embeddings = frame_embeddings.view(batch_size, N, -1) #This will result in a shape [batch_size, N, vid_embd_dim]
        video_embedding = frame_embeddings.mean(dim=1) #This will result in a shape [batch_size, vid_embd_dim] (average over N)
        #Lastly, project the video_embeddings into the smaller space
        video_embedding = self.video_proj(video_embedding) #This will result in a shape [batch_size, aggregated_dim]
        
        #For the audio and text data, we need to iterate over all of the elements in the batch
        #Start by iterating through each of the examples in the batch
        batch_audio_embeddings = []
        batch_text_embeddings = []
        for i in range(len(batch_text)):
            #-------------------------------
            #------- Audio Embedding -------
            #-------------------------------
            audio = batch_audio[i]
            #Move the features to the GPU
            audio = {k: v.to(self.device) for k, v in audio.items()}
            #Pass the audio data through the HuBERT model (the audio data must already be processed)
            #print(f"Audio input shape before HuBERT: {audio['input_values'].shape}")
            audio_model_out = self.audio_encoder(**audio) #This passes the key-word arguements of ths audio data
            #Pull the final embedding vector out of this model
            audio_embedding = audio_model_out.last_hidden_state #Shape [batch_size, time_len, aud_embd_dim]
            #Average over the time dimension
            audio_embedding = audio_embedding.mean(dim=1) #Shape [batch_size, aud_embd_dim]
            #Lastly, project the video_embeddings into the smaller space
            audio_embedding = self.audio_proj(audio_embedding) #Shape [batch_size, aggregated_dim]
            batch_audio_embeddings.append(audio_embedding[0])
            
            #-------------------------------
            #-------- Text Embedding -------
            #-------------------------------
            # The text data is a list of dicts for each caption (see the text_transforms function below),
            # e.g. [ { 'input_ids': tensor(...), 'attention_mask': tensor(...) },
            #        { 'input_ids': tensor(...), 'attention_mask': tensor(...) }, ... ]
            embedding_list = []
            text = batch_text[i]
            for tokenized_cap in text: #tokenized_cap will have shape [1, seq_len]
                #Move the tokens to the GPU or CPU
                tokenized_cap = {k: v.to(self.device) for k, v in tokenized_cap.items()}
                #Pass the text data through the BERT model
                model_embeddings = self.text_encoder(**tokenized_cap)
                #Pull the final embedding vector out of this model
                text_embedding = model_embeddings.pooler_output
                embedding_list.append(text_embedding)
            #Aggregate the caption embeddings
            if len(embedding_list) > 1:
                text_embedding = torch.mean(torch.cat(embedding_list, dim=0), dim=0, keepdim=True).to(self.device)
            else:
                text_embedding = embedding_list[0]
            #Lastly, project the video_embeddings into the smaller space
            text_embedding = self.text_proj(text_embedding) #Shape [batch_shape, aggregated_dim]
            batch_text_embeddings.append(text_embedding[0])
        batch_audio_embeddings = torch.stack(batch_audio_embeddings, dim=0).to(self.device)
        batch_text_embeddings = torch.stack(batch_text_embeddings, dim=0).to(self.device)
        
        #-------------------------------
        #--------- Fusion & MLP --------
        #-------------------------------
        #Concatenate the embeddings
        fused_embedding = torch.cat([video_embedding, batch_audio_embeddings, batch_text_embeddings], dim=1).to(self.device)
        #Pass through the MLP to generate prediction
        final_embedding = self.mlp(fused_embedding)
        
        return final_embedding


##############################################################################################################################################################
##################################################################### Define the Dataset #####################################################################
##############################################################################################################################################################
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class VideoDataSet(Dataset):
    """
    This class defines the video dataset
    """
    def __init__(self, video_path, annotation_path, num_frame_samples=10, audio_samp_rate=16000, titles=None, vid_trans=None, aud_trans=None, tex_trans=None):
        super().__init__()
        self.video_path = video_path
        self.annotation_path = annotation_path
        self.num_frame_samples = num_frame_samples
        self.audio_samp_rate = audio_samp_rate
        self.titles = titles
        self.vid_trans = vid_trans #Video transformations
        self.aud_trans = aud_trans #Audio transformations
        self.tex_trans = tex_trans #Text transformations
        
        #Load the .json annotations file
        with open(annotation_path, 'r') as file:
            self.annotations = json.load(file)
        
        #Make a dictionary with video titles as keys and each element a list of all of the captions for that video
        #This should result in a dictionary with one key for each video
        self.titles_set = set(self.titles)
        if self.titles:
            #This means that we only want the videos corresponding to those on the list of titles
            self.video_caps = defaultdict(list)
            for i in range(len(self.annotations['annotations'])):
                vid_title = self.annotations['annotations'][i]['image_id']
                if vid_title in self.titles_set:
                    caption_i = self.annotations['annotations'][i]['caption']
                    self.video_caps[vid_title].append(caption_i)
        else:
            #This means that we want all of the videos (all 10,000)
            self.video_caps = defaultdict(list)
            for i in range(len(self.annotations['annotations'])):
                vid_title = self.annotations['annotations'][i]['image_id']
                caption_i = self.annotations['annotations'][i]['caption']
                self.video_caps[vid_title].append(caption_i)
    
    def __len__(self):
        return len(self.video_caps)
    
    def __getitem__(self, index):
        #The title of the video
        if self.titles:
            video_title = self.titles[index]
        else:
            video_title = self.annotations['images'][index]['id'] 
        captions = self.video_caps[video_title] #The list of all the captions of the video
        #Path to the .mp4 file for the video
        video_file = os.path.join(self.video_path, f'{video_title}.mp4')
        #Read the video and the audio
        frames, audio_wave = self._load_vid_and_aud(video_file)
        #Pad the wave if it is too short in length to be given to HuBERT
        audio_wave = pad_waveform(audio_wave)
        #Apply transformations if there are any
        if self.vid_trans:
            frames = torch.stack([self.vid_trans(image) for image in frames], dim=0)
        if self.aud_trans:
            audio_wave = self.aud_trans(audio_wave, self.audio_samp_rate)
        if self.tex_trans:
            captions = self.tex_trans(captions, None)
        return frames, audio_wave, captions, video_title

    def _load_vid_and_aud(self, path):
        """
        This is a function that uses PyAV to load the sample frames and audio from the .mp4 file at the path address.
        This will return the sample frames (a list of tensors), the raw audio waveform, and the sample rate.
        """
        container = av.open(path)
        #----- Extract the video frames -----
        video_stream = container.streams.video[0]
        frames_list = []
        total_frames = video_stream.frames
        sampled_indices = self._sample_frame_indices(total_frames, self.num_frame_samples) #the indices of the frames that we will sample
        #Loop over container.decode(video=0)
        decoded_frames = []
        for i, frame in enumerate(container.decode(video=0)):
            #Only keep the sampled frames
            if i in sampled_indices:
                image = frame.to_image() #Converts the AVFrame to a PIL image
                #Convert the PIL image to a PyTorch tensor
                torch_image = torch.as_tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())), dtype=torch.uint8)
                torch_image = torch_image.view(image.height, image.width, 3).permute(2,0,1).float() #Returns shape [3, H, W]
                decoded_frames.append(torch_image)
            if len(decoded_frames) == self.num_frame_samples:
                break
        #Include edge case?
        frames_list = decoded_frames #This is a list of frames, each of shape [3, H, W]
        
        #----- Extract the audio frames -----
        container.seek(0)
        audio_wave = torch.zeros(1)
        samp_rate = self.audio_samp_rate
        audio_streams = [stream for stream in container.streams if stream.type == 'audio']
        if len(audio_streams) > 0:
            audio_stream = audio_streams[0]
            raw_audio = []
            for frame in container.decode(audio = 0):
                audio_array = frame.to_ndarray() #Shape (samples, channels])
                samples = torch.from_numpy(audio_array).float() #Shape [samples, channels]
                raw_audio.append(samples)
            if len(raw_audio) > 0:
                audio_wave = torch.cat(raw_audio, dim=1).float() #Shape [total_samples * channels]
                ch = audio_stream.channels if audio_stream.channels else 2 #The number of channels
                audio_wave = audio_wave.view(-1, ch).t() #Shape [channels, total_samples]
                #determine the sample rate
                orig_samp_rate = audio_stream.codec_context.sample_rate
                #Resample to the desired sample rate
                if orig_samp_rate != self.audio_samp_rate:
                    audio_wave = torchaudio.functional.resample(audio_wave, orig_freq=orig_samp_rate, new_freq=samp_rate)
        else:
            audio_wave = torch.zeros([1]) #This is what will be used if the video has no audio
        container.close()
        #Turn the lists into tensors
        frames_list = torch.stack(frames_list, dim=0)
        return frames_list, audio_wave

    def _sample_frame_indices(self, total_frames, num_frames):
        """
        This is a function that uniformly samples the desired number of frame indices from a video.
        If the 'num_frames' is None, then all the frames will be utilized.
        """
        if num_frames is None or num_frames >= total_frames:
            return list(range(total_frames))
        #Uniformly sample the frame indices
        dist_between_frames = total_frames / num_frames
        indices = [int(i*dist_between_frames) for i in range(num_frames)]
        return indices

def pad_waveform(audio_wave, min_length=5000):
    #First, make sure that the wave has at least a 2D shape. If not, give it 2D shape
    if audio_wave.ndim == 1:
        audio_wave = audio_wave.unsqueeze(0)
    num_samples = audio_wave.shape[-1] #This is the current length of the audio wave
    if num_samples < min_length:
        pad_length = min_length - num_samples
        audio_wave = torch.cat([audio_wave, torch.zeros((audio_wave.shape[0], pad_length), device=audio_wave.device)], dim=-1)
    return audio_wave


def collate_fn(batch):
    """
    This is a custom collate function which will be called by the DataLoader. This will handle variable numbers of frames.
    If all of the frames have the same shape, then we can stack them.
    
    Parameters
    ----------
    batch : A list of (frames, audio_waves, captions)
    """
    frame_list, audio_list, text_list, title_list = [], [], [], []
    for frames, audio_wave, text, title in batch:
        frame_list.append(frames)
        audio_list.append(audio_wave)
        text_list.append(text)
        title_list.append(title)
    #Convert the list of frames to a tensor
    frames_torch = torch.stack(frame_list, dim=0)
    return frames_torch, audio_list, text_list, title_list

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
################################################################## Modality Transformations ##################################################################
##############################################################################################################################################################
import torchvision.transforms as T

#Video transforms function
def video_transforms(frames):
    """
    This simply takes the input and reshapes and normalizes it to fit for ResNet.
    """
    vid_transforms = T.Compose([
        T.Resize((224,224)), #Reshape for ResNet
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ])
    return vid_transforms(frames)

#Audio transforms function
def audio_transforms(audio_wave, samp_rate):
    #If there is more than one channel, average them to get a mono wave
    if audio_wave.shape[0] > 1:
        audio_wave = audio_wave.mean(dim=0, keepdim=True) #This will return shape [1, total_samples]
    #Normalize the waveform
    max_wave_amp = audio_wave.abs().max()
    if max_wave_amp > 0:
        audio_wave = audio_wave / max_wave_amp
    #Now, utilize the HuBERT feature extractor, which is a Wav2VecFeatureExtractor under the hood which is specialized for HuBERT
    #(This is similar to using the BERT tokenizer for the text data)
    processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    audio_features = processor(audio_wave, sampling_rate=samp_rate, return_tensors="pt") #A dictionary with 'input_values' having a torch tensor with shape [1,1,1,total_samples]
    #Reshape the tensor so that HuBERT will like it
    audio_features['input_values'] = audio_features['input_values'].reshape(1, audio_features['input_values'].shape[-1]) #Shape [1, total_samples]
    return audio_features #This is a dictionary, with 'input_ids', 'attention_mask', etc. as keys
    
#Text transforms function
def text_transforms(captions, n_samples):
    """
    This simply takes the input (a list of captions for a particular video), utilizes the pre-trained ALBERT auto-tokenizer
    to generate tokenizations for each of them.
    """
    if n_samples == None: #This indicates that n_samples is None, and we will use all of the captions
        tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2") #This is what we will use to turn each caption into a tokenization
        tokenized_list = []
        for cap in captions:
            #Each encoding is a dict with 'input_ids', 'attention_mask', etc. and each sample is shape [1, seq_len] because return_tensors="pt" and a single string
            encoding = tokenizer(cap, padding="max_length", truncation=True, max_length=len(cap), return_tensors="pt")
            tokenized_list.append(encoding)
    else: #This means that n_samples is a numeric value and we will sample n_samples of the captions (already tokenized)
          #(this will help generate augmentations for the text embeddings in the contrastive setting)
          #In this case, captions will be a list lists (a batch of captions for different videos)
          tokenized_list = []  
          for i in range(len(captions)):
                if n_samples < len(captions[i]):
                    tokenized_samples = random.sample(captions[i], n_samples)
                    tokenized_list.append(tokenized_samples)
          return tokenized_list #This is a list of dictionaries, as described above
    return tokenized_list #This is a list of dictionaries, as described above


##############################################################################################################################################################
################################################################# Contrastive Transformations ################################################################
##############################################################################################################################################################
import torchaudio.transforms as AT

#Contrastive transformation for videos
def video_transforms_contrast(frames):
    """
    Performs augmentations for contrastive loss
    """
    vid_transforms = T.Compose([
        T.RandomResizedCrop(200, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ])
    batch_size, N, C, H, W = frames.shape
    #Apply transforms to each frame independently
    frames = frames.view(-1, C, H, W) #Resape to [batch_size * N, C, H, W]
    #Apply transforms frame-wise
    frames = torch.stack([vid_transforms(frame) for frame in frames])
    frames = frames.view(batch_size, N, C, 200, 200) #Reshape to original shape (with 224 replaced with 200 since we are using RandomResizedCrop)
    return frames

#Contrastive transformation for audio
def audio_transforms_contrast(audio_wave_list):
    """
    Performs augmentations for contrastive loss
    """
    transformed_audio_list = []
    for i in range(len(audio_wave_list)):
        wave = audio_wave_list[i]['input_values'].clone()
        #Reduce volume
        wave = AT.Vol(0.5)(wave)
        #Add some Gaussian noise
        noise = torch.randn_like(wave) * 0.1
        wave = wave + noise
        transformed_audio_list.append({'input_values': wave})
    return transformed_audio_list

#Notice that for the text transformations, we can simply use the original
#text_transforms function, but now just pass in a sample size n_samples


##############################################################################################################################################################
################################################################## Contrastive Loss Function #################################################################
##############################################################################################################################################################
class ContrastiveLoss(nn.Module):
    """
    This is a cross-entropy contrastive loss function for batches of embeddings.
    This is a self-supervised learning loss which works by moving positive examples
    closer to each other and away from negative examples. It requires a query embedding
    denoted by embd_A and a positive example embd_B.
    Both of the embedding vectors are of shape [batch_size, embd_dim].
    """
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp
    
    def forward(self, embd_A, embd_B):
        #Normalize for stable dot-products
        embd_A = F.normalize(embd_A, dim=1)
        embd_B = F.normalize(embd_B, dim=1)
        #Compute the similarity matrix (This is a matrix of similarity scores between all of the samples in each batch)
        G = torch.matmul(embd_A, embd_B.t()) / self.temp #Shape [batch_size, batch_size]
        G_T = G.t() #The transpose of G
        #The diagonal elemenst G[i, i] is the similarity between embd_A[i] and embd_B[i] (the correct match)
        #Similarly, the off-diagonals G[i, j] (i not = j) are negative pairs (and should be pushed apart).
        target = torch.arange(embd_A.size(0), device=embd_A.device) #This creates a tensor list [0, 1, 2, ..., batch_size - 1] which serves as the ground-truth labels (indexes)
        loss_A = F.cross_entropy(G, target) #The error in how well embedding A matches embedding B
        loss_B = F.cross_entropy(G_T, target) #The error in how well embedding B matches embedding A
        return (loss_A + loss_B) * 0.5 #Return the average loss
        
        
##############################################################################################################################################################
###################################################################### Training Algorithm ####################################################################
##############################################################################################################################################################
def train_algo(model, dataloaders, contrast_transforms, n_cap_samples, optimizer, loss_func, device, num_epochs, lr_scheduler, sched_type):
    """
    Training algorithm
    """
    #Set up an interactive plot
    plt.ion()
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    line, = ax.plot(xdata, ydata, '-')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Contrastive Cross-Entropy Loss')
    ax.set_title('Training Multi-Net Model for Video Recommendation \n via Transfer Learning with Contrastive Loss', size=12)
    ax.grid(True)
    
    #Pull out the transforms to generate contrastive examples
    vid_trans_contrast, aud_trans_contrast, tex_trans_contrast = contrast_transforms[0], contrast_transforms[1], contrast_transforms[2]
    
    train_dataloader = dataloaders[0]
    val_dataloader = dataloaders[1]
    iter_list = []
    loss_list = []
    val_loss_list = []
    i = 0
    for epoch in range(num_epochs):
        epoch_error_list = []
        for (f, a, t, _) in train_dataloader:
            model.train()
            iter_list.append(i)
            i+=1
            
            #Generate positive example embeddings for contrastive learning
            f_contrast = vid_trans_contrast(f)
            a_contrast = aud_trans_contrast(a)
            t_contrast = tex_trans_contrast(t, n_cap_samples)
            
            f = f.to(device); f_contrast = f_contrast.to(device)
            #Model prediction
            embedding_A = model(f, a, t)
            embedding_B = model(f_contrast, a_contrast, t_contrast)
            loss = loss_func(embedding_A, embedding_B)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            epoch_error_list.append(loss.item())
            
            if sched_type == "one_cycle":
                lr_scheduler.step()
            
            #Compute the validation loss
            if val_dataloader:
                model.eval()
                with torch.no_grad():
                    f_val, a_val, t_val, _ = next(iter(val_dataloader))
                    #Generate positive example embeddings for contrastive learning
                    f_contrast_val = vid_trans_contrast(f_val)
                    a_contrast_val = aud_trans_contrast(a_val)
                    t_contrast_val = tex_trans_contrast(t_val, n_cap_samples)
                    f_val = f_val.to(device); f_contrast_val = f_contrast_val.to(device)
                    #Model prediction
                    val_embedding_A = model(f_val, a_val, t_val)
                    val_embedding_B = model(f_contrast_val, a_contrast_val, t_contrast_val)
                    val_loss = loss_func(val_embedding_A, val_embedding_B)
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
################################################# Use a trained Model to Generate Embeddings for All Videos ##################################################
##############################################################################################################################################################
def generate_learned_embeddings(dataloader, model, device):
    """
    This function takes a trained model and a dataloader and returns a table of all video embeddings.
    This table of embeddings can now be use to serve up recommendations.
    """
    model.to(device)
    model.eval()
    embedding_lookup_table = {}
    i = 0
    for (f, a, t, title) in dataloader:
        print(i)
        i+=1
        f = f.to(device)
        video_title = title
        embedding = model(f, a, t)
        embedding_lookup_table[video_title[0]] = embedding.cpu().detach().numpy()
    return embedding_lookup_table


##############################################################################################################################################################
################################################################# K-Nearest Neighbor Service #################################################################
##############################################################################################################################################################
def knn_server(k, query_video, embedding_lookup_table):
    """
    This function takes the table of final video embeddings and returns the top K videos nearest to the query video.
    """
    query_embedding = embedding_lookup_table[query_video]
    #Compute the Euclidean distances
    distances = {title: np.linalg.norm(embd - query_embedding) for title, embd in embedding_lookup_table.items()}
    #Top K nearest embeddings
    top_k = sorted(distances.items(), key=lambda x: x[1])[:k]
    #Names for the top K embeddings
    top_k_names = [name for name, vec in top_k]
    return top_k_names


##############################################################################################################################################################
############################################################################ Main ############################################################################
##############################################################################################################################################################
def main():
    ###########################################################################
    #################### Define Parameters & Load Dataset #####################
    ###########################################################################
    video_path = "INSERT PATH HERE"
    annotation_path = "INSERT PATH HERE"
    train_batch_size = 6
    test_batch_size = 1
    num_frame_samples = 10 #15
    audio_samp_rate=16000
    vid_trans = video_transforms
    aud_trans = audio_transforms
    tex_trans = text_transforms
    Set_Seed(42)
    
    #Load the list of video names to be used for training
    training_path = "INSERT PATH HERE"
    with open(training_path, 'r') as file:
        train_vid_titles = file.read()
    train_vid_titles = train_vid_titles.split("\n")
    #Load the list of video names to be used for training
    test_path = "INSERT PATH HERE"
    with open(test_path, 'r') as file:
        test_vid_titles = file.read()
    test_vid_titles = test_vid_titles.split("\n")
    
    #Create the training dataset (7,010 videos) and the corresponding dataloader
    train_dataset = VideoDataSet(video_path, annotation_path, num_frame_samples, audio_samp_rate, train_vid_titles, vid_trans=vid_trans, aud_trans=aud_trans, tex_trans=tex_trans)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    #Create the testing dataset (2,990 videos) and the corresponding dataloader
    test_dataset = VideoDataSet(video_path, annotation_path, num_frame_samples, audio_samp_rate, test_vid_titles, vid_trans=vid_trans, aud_trans=aud_trans, tex_trans=tex_trans)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = None #Only do this if we don't want to pass a testing dataset
    dataloaders = [train_dataloader, test_dataloader]
    
    ###########################################################################
    ########################### Model and Training ############################
    ###########################################################################
    #Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 3e-6 #1e-6
    num_epochs = 2 #5
    aggregated_dim = 100 #This is the dimension size that each of the modality embeddings will be mapped to before they are concatenated
    contrast_transforms = [video_transforms_contrast, audio_transforms_contrast, text_transforms] #List of functions for the contrast augmentations
    n_cap_samples = 5 #Number of captions to sample for the contrastive augmentations
    
    #Define the model
    model = MultiModalModel(aggregated_dim, device)
    model.to(device)
    #Define loss, optimizer, and scheduler
    loss_func = ContrastiveLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, epochs=num_epochs, steps_per_epoch=len(train_dataloader))
    sched_type = "one_cycle" #"none" #"one_cycle" #"plateau"
    
    #Train the model
    iter_list, train_loss_list, val_loss_list = train_algo(model, dataloaders, contrast_transforms, n_cap_samples, optimizer, loss_func, device, num_epochs, lr_scheduler, sched_type)
    #Plotting
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    sns.lineplot(x=iter_list, y=train_loss_list, linewidth=0.3, label='Training Loss', legend=True, color='blue', linestyle='-').set_title('Trained Multi-Net Model for Video Recommendation \n via Transfer Learning with Contrastive Loss', size=15)
    #sns.lineplot(x=iter_list, y=val_loss_list, linewidth=0.3, label='Validation Loss', legend=True, color='red', linestyle='-')
    plt.xlabel("Iterations")
    plt.ylabel("Contrastive Cross-Entropy Loss")
    axes.grid(True)
    plt.show()
    
    
    ###########################################################################
    ########################### Model Save / Import ###########################
    ###########################################################################
    model_path = "INSERT PATH HERE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregated_dim = 100
    #Save the model
    torch.save(model.state_dict(), model_path)
    #Or run this if you want to load the model
    model = MultiModalModel(aggregated_dim, device)
    model.load_state_dict(torch.load(model_path))
    

    ###########################################################################
    ####### Create a Table of video Embeddings using the Trained Model ########
    ###########################################################################
    #Define the dataloader that we will embed
    video_path = "INSERT PATH HERE"
    annotation_path = "INSERT PATH HERE"
    train_batch_size = 1
    num_frame_samples = 16
    audio_samp_rate=16000
    vid_trans = video_transforms
    aud_trans = audio_transforms
    tex_trans = text_transforms
    Set_Seed(42)
    #Load the list of video names to be used for training
    training_path = "INSERT PATH HERE"
    with open(training_path, 'r') as file:
        train_vid_titles = file.read()
    train_vid_titles = train_vid_titles.split("\n")
    #Create the training dataset (7,010 videos) and the corresponding dataloader
    train_dataset = VideoDataSet(video_path, annotation_path, num_frame_samples, audio_samp_rate, train_vid_titles, vid_trans=vid_trans, aud_trans=aud_trans, tex_trans=tex_trans)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn)
    #Embed all videos with the trained model
    embedding_lookup_table = generate_learned_embeddings(train_dataloader, model, device)
    #Save the table of embeddings
    import pickle
    path = "INSERT PATH HERE"
    with open(path, "wb") as f:
        pickle.dump(embedding_lookup_table, f)
    
    
    ###########################################################################
    ###################### Load Embedding Look-Up Table #######################
    ###########################################################################
    import pickle
    path = "INSERT PATH HERE"
    with open(path, "rb") as f:
        embedding_lookup_table = pickle.load(f)
    
    
    ###########################################################################
    ###################### Load Embedding Loop-Up Table #######################
    ###########################################################################
    #Define the query video
    query_video = 'video47'
    #Return the top k most similar videos
    k = 5
    top_k_titles = knn_server(k, query_video, embedding_lookup_table)
    print(f"Top K={k} video recommendations similar to video {query_video}: "+str(top_k_titles))
    



if __name__ == "__main__":
    main()

