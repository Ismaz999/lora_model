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

test_resnet = Subset(dataset, indices=list(range(1000)))
input_data = DataLoader(test_resnet, batch_size=32, shuffle=False, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_features.to(device)

features = []

with torch.no_grad():
    for batch_images, batch_captions in input_data:
        batch_images = batch_images.to(device)
        batch_outputs = model_features(batch_images)

        batch_outputs = batch_outputs.mean(dim=[2, 3])
        features.extend(batch_outputs)


# Définition du modèle de décodage

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, vocab):
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(2048 + embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    def forward(self, features, captions):

        features_avg = features.mean(dim=[2, 3])

        embeddings = self.dropout(self.embed(captions)).permute(1, 0, 2)
        
        features_repeated = features_avg.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        
        embeddings = torch.cat((features_repeated, embeddings), dim=2)
        
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
        
features_tensor = torch.stack(features)

captions_transformed = [transform_caption(caption, vocab) for caption in image_annot[:len(features)]]

captions_tensor = torch.tensor(captions_transformed)

EMBED_SIZE = 256
HIDDEN_SIZE = 512

decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size, vocab).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(decoder.parameters(), lr=0.001)

# Paramètres d'entraînement
num_epochs = 10
print_every = 10

# Boucle d'entraînement
for epoch in range(num_epochs):
    
    for i, (images, captions) in enumerate(input_data):
        # Mettre les images et les légendes sur le périphérique approprié
        images = images.to(device)

        # print(type(captions))
        # print(len(captions))
        # if isinstance(captions, list):
        #     for item in captions:
        #         print(type(item), len(item))

        captions_tensor = torch.stack(captions).to(device)

        # Récupérer les features des images à l'aide du modèle ResNet pré-entraîné
        with torch.no_grad():
            features = model_features(images)

        # Initialiser les gradients du décodeur à zéro
        decoder.zero_grad()

        outputs = decoder(features, captions_tensor)

        loss = criterion(outputs.view(-1, vocab_size), captions_tensor.view(-1))

        # Rétropropagation
        loss.backward()

        # Mise à jour des poids
        optimizer.step()

        if i % print_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {i}/{len(input_data)}, Loss: {loss.item()}")




