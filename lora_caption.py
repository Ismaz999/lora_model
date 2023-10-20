import os

#torch related imports
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, Subset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

#nltk imports
import nltk
from nltk.tokenize import word_tokenize

#other imports
import numpy as np
from PIL import Image

#project imports
from data_utils import transform_img, CaptionDataset, build_vocab, transform_caption
from RNN_decoders import DecoderRNN, DecoderRNN2
from model_github import EncoderCNN3, DecoderRNN3

#global config variables
LOAD_PRETRAINED = False
NumModel = 2
SAMPLE_SIZE = 50

### Program ###
###############

# nltk.download('punkt')

# Chargement des données
file = open('captions.txt', 'r')
file_content = file.readlines()
file_content = file_content[1:]

image_dict = {}
image_path = []
image_annot = []

for ligne in file_content:
    parts = ligne.strip().split(',')
    path = parts[0]
    annot = ','.join(parts[1:]).strip().rstrip('.')

    image_path.append(path)
    image_annot.append(annot)
    image_dict[path] = annot

root = "C:\\Users\\Audensiel\\Desktop\\piccture_anotation\\Images"

# Construction du vocabulaire
vocab = build_vocab(image_annot)
# print("le vocab:", vocab)
vocab_size = len(vocab)

if '<unk>' not in vocab:
    vocab['<unk>'] = len(vocab)


dataset = CaptionDataset(image_annot, image_path, root, vocab=vocab, transform=transform_img)

# Prétraitement des données

resnet_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
model_features = torch.nn.Sequential(*list(resnet_model.children())[:-2])
model_features.eval()

test_resnet = Subset(dataset, indices=list(range(SAMPLE_SIZE)))
input_data = DataLoader(test_resnet, batch_size=16, shuffle=False, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_features.to(device)

features = []

with torch.no_grad():
    for batch_images, batch_captions in input_data:
        batch_images = batch_images.to(device)
        #print("batch_images dimension:", batch_images.shape)
        batch_outputs = model_features(batch_images)
        #print("batch_outputs dimensions:", batch_outputs.shape)
        batch_outputs = batch_outputs.mean(dim=[2, 3])
        features.extend(batch_outputs)

#### Analyse des dimensions ####
#
#   batch_images = [16, 3, 256, 256]
#      16 = nombre d'images dans le batch, 3 pour RGB, [256,256] pour la dimensions des images
#
#   batch_outputs = [16, 2048, 8, 8]
#       16 = nombre de données dans le batch, 2048 output de la couche des features, [8,8] car les images ont été réduites a 8x8
#
################################

features_tensor = torch.stack(features)

captions_transformed = [transform_caption(caption, vocab) for caption in image_annot[:len(features)]]

captions_tensor = torch.tensor(captions_transformed)

EMBED_SIZE = [512]
HIDDEN_SIZE = [1024]
LEARNING_RATE = [0.001]
WEIGHT_DECAYS = [1e-5]

best_loss = float('inf')
best_params = {}

# decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size, vocab).to(device)
# criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
# optimizer = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=1e-5)


# Paramètres d'entraînement
num_epochs = 10
print_every = 10


if LOAD_PRETRAINED:
    model_save_path = "best_decoder_model_NumModel"+str(NumModel)+".pth"
    
    checkpoint = torch.load(model_save_path)
    best_params = checkpoint['best_params']
    if NumModel == 1:
        loaded_model = DecoderRNN(embed_size=best_params["embedding_size"], 
                                  hidden_size=best_params["hidden_size"], 
                                  vocab_size=vocab_size, vocab=vocab).to(device)
    elif NumModel == 2:
        loaded_model = DecoderRNN2(embed_size=best_params["embedding_size"], 
                                hidden_size=best_params["hidden_size"], 
                                vocab_size=vocab_size).to(device)
    
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    print("Modèle chargé depuis", model_save_path)

# Boucle d'entraînement
if not LOAD_PRETRAINED:
    for lr in LEARNING_RATE:
        for embed_size in EMBED_SIZE:
            for hidden_size in HIDDEN_SIZE:
                for weight_decay in WEIGHT_DECAYS:
                    
                    if NumModel == 1:
                        decoder = DecoderRNN(embed_size, hidden_size, vocab_size, vocab).to(device)
                    elif NumModel == 2:
                        decoder = DecoderRNN2(embed_size, hidden_size, vocab_size).to(device)
                    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
                    optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)

                    total_loss = 0
                    for epoch in range(num_epochs):
                        for i, (images, captions) in enumerate(input_data):
                            images = images.to(device)
                            #print("images dimension:", images.shape)
                            captions_tensor = torch.stack(captions).to(device)
                            #print("captions_tensor dimension:", captions_tensor.shape)
                            
                            #### Analyse des dimensions ####
                            #
                            #   images = [16, 3, 256, 256]
                            #      16 = nombre d'images dans le batch, 3 pour RGB, [256,256] pour la dimensions des images
                            #
                            #   caption_tensor = [50, 16]
                            #       50 = Longueur max des annotations, 16 pour les images du lot auquels sont associés des annotations  
                            #
                            ################################



                            with torch.no_grad():
                                features = model_features(images)
                            
                            decoder.zero_grad()
                            outputs = decoder(features, captions_tensor)
                            loss = criterion(outputs.view(-1, vocab_size), captions_tensor.view(-1))
                            
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()

                    avg_loss = total_loss / (num_epochs * len(input_data))
                    print(f"LR: {lr}, Embed Size: {embed_size}, Hidden Size: {hidden_size}, Weight Decay: {weight_decay}, Avg Loss: {avg_loss}")

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_params = {
                            "learning_rate": lr,
                            "embedding_size": embed_size,
                            "hidden_size": hidden_size,
                            "weight_decay": weight_decay
                        }
                        best_decoder = decoder
                    
    print("Meilleurs paramètres:", best_params)
    model_save_path = "best_decoder_model_NumModel"+str(NumModel)+".pth"
    torch.save({
    'model_state_dict': best_decoder.state_dict(),
    'best_params': best_params
}, model_save_path)
    print(f"Modèle sauvegardé à {model_save_path}")


# Initialiser le modèle avec les meilleurs paramètres
if NumModel == 1:
    decoder = DecoderRNN(best_params["embedding_size"], best_params["hidden_size"], vocab_size, vocab).to(device)
elif NumModel == 2:
    decoder = DecoderRNN2(best_params["embedding_size"], best_params["hidden_size"], vocab_size).to(device)

# Charger les poids sauvegardés
decoder.load_state_dict(torch.load("best_decoder_model_NumModel"+str(NumModel)+".pth")['model_state_dict'])
decoder.eval()  # Mettre le modèle en mode évaluation


def generate_caption(decoder, image_feature, vocab):

    start_token = vocab["<start>"]
    end_token = vocab["<end>"]
    
    inputs = torch.LongTensor([start_token]).unsqueeze(0).to(device)
    caption = []
    
    with torch.no_grad():
        for _ in range(50):  # 50 est la longueur maximale de la légende
            outputs = decoder(image_feature.unsqueeze(0), inputs)
            _, predicted = outputs.max(2)
            next_word = predicted[0, -1].item()
            
            if next_word == end_token:
                break
            
            caption.append(next_word)
            inputs = torch.cat([inputs, torch.LongTensor([next_word]).unsqueeze(0).to(device)], dim=1)

    words = [word for idx, word in vocab.items() if idx in caption]
    return " ".join(words)

def show_image_with_caption(image_tensor, caption):
    """Affiche une image avec sa légende."""
    image = image_tensor.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # plt.imshow(image)
    print("caption: ",caption)
    # plt.savefig("C:\\Users\\Audensiel\\Desktop\\piccture_anotation\\"+caption+".png")
    # plt.show()


# Testons sur quelques images
num_samples = 5

for i, (image, _) in enumerate(input_data):
    with torch.no_grad():
        feature = model_features(image.to(device))
    
    print("image_features", feature[0].shape)
    print("image_features 2", feature[0].unsqueeze(0).shape)
    #dimension de feature[0] 2048,8,8 car 2048 pour l'output de la couche feature de resnet et 8x8 pour la représentation spatiale des features, unsqueeze(0) ajoute 1 dimension
    caption = generate_caption(decoder, feature[0], vocab)
    show_image_with_caption(image[0], caption)

    if i == 2:
        break


