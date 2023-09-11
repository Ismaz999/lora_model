import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, Subset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from data_utils import transform_img, CaptionDataset, build_vocab, transform_caption

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

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

dataset = CaptionDataset(image_annot, image_path, root, transform=transform_img)

# Prétraitement des données

resnet_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
model_features = torch.nn.Sequential(*list(resnet_model.children())[:-2])
model_features.eval()

test_resnet = Subset(dataset, indices=list(range(50)))
input_data = DataLoader(test_resnet, batch_size=32, shuffle=False, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_features.to(device)

features = []

with torch.no_grad():
    for batch_images, batch_captions in input_data:
        batch_images = batch_images.to(device)
        batch_outputs = model_features(batch_images)

        features.extend(batch_outputs)

# for images, tokens in input_data:
#     print("Tokens:", tokens)

# print("LIST DANNOT",image_annot)


# Définition du modèle de décodage

class DecoderRNN(nn.Module):
    def __init__(self, feature, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.lstm = nn.LSTM(self.embed_size, 
                            self.hidden_size, 
                            self.num_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.hidden = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

    def forward(self, features, captions):
        # Embeddings des mots
        captions = self.word_embedding(captions)
        
        # Concaténer les features de l'encodeur avec les embeddings des mots
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        
        # Passer à travers le LSTM
        out, _ = self.lstm(inputs)
        
        # Passer à travers la couche de sortie
        out = self.fc(out)
        
        return out
        
captions = image_annot
vocab = build_vocab(captions)

if '<unk>' not in vocab:
    vocab['<unk>'] = len(vocab)

indices = [transform_caption(caption, vocab) for caption in captions]

all_tokens = []

for indices_par_images in indices:
    all_tokens.extend(indices_par_images)
                    
tokens = torch.tensor(all_tokens).to(device)

embed_size = 256
hidden_size = 512
vocab_size = len(vocab)

decoder = DecoderRNN(features, embed_size, hidden_size, vocab_size)

# Définition de la fonction de perte (entropie croisée) pour l'entraînement
criterion = nn.CrossEntropyLoss()

# Optimiseur pour la mise à jour des poids du décodeur
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)  # Vous pouvez ajuster le taux d'apprentissage (learning rate) si nécessaire

# Nombre d'époques d'entraînement
num_epochs = 10  # Vous pouvez ajuster ce nombre en fonction de la convergence

decoder.to(device)  # Transférer le décodeur sur le GPU si disponible

for epoch in range(num_epochs):
    for batch_images, batch_tokens in input_data:
        # Transférez les données sur le GPU si disponible
        batch_images = batch_images.to(device)   

        batch_indices = [transform_caption(' '.join(caption), vocab) for caption in batch_captions]

        tensor_batch_captions = torch.LongTensor(batch_indices)
        tensor_batch_captions = tensor_batch_captions.to(device)

        # Réinitialisez les gradients
        decoder_optimizer.zero_grad()
        
        # Passez l'image à travers l'encodeur pour obtenir les caractéristiques
        features = model_features(batch_images)
        
        # Passez les caractéristiques de l'image et les légendes au décodeur
        batch_tokens_tensor = torch.tensor(batch_tokens).to(device)
        outputs = decoder(features, batch_tokens_tensor)
        
        # Calculez la perte en comparant les prédictions aux légendes réelles
        loss = criterion(outputs.view(-1, vocab_size), batch_tokens.view(-1))  # Aplatissez les tenseurs
        
        # Effectuez une rétropropagation pour mettre à jour les poids du décodeur
        loss.backward()
        decoder_optimizer.step()
        
    # Affichez la perte à la fin de chaque époque
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')