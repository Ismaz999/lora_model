from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
from torchvision.transforms import ToTensor
from nltk.tokenize import word_tokenize



transform_img = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

def build_vocab(captions):
    vocab = {'<start>': 0, '<end>': 1, '<pad>': 2}
    
    for caption in captions:
        tokens = caption.split()
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    return vocab

def transform_caption(caption, vocab, max_length=50):
    tokens = caption.split()  
    # print("voici les tokens", tokens)
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    while len(indices) < max_length:
        indices.append(vocab["<pad>"])
    return indices[:max_length]

class CaptionDataset(torch.utils.data.Dataset):
   def __init__(self, caption_array, image_path, root, vocab, transform=None, max_token_length=50): #Permet d'initialiser les variables avant d'accéder a un élément précis du dataset

      self.caption_array = caption_array
      self.root = root
      self.transform = transform
      self.image_path = image_path
      self.max_token_length = max_token_length
      self.vocab = vocab

   def __getitem__(self, index): #Permet d'accéder à un élément spécifique du dataset
      annotation = self.caption_array[index]
      image_name = self.image_path[index]
      
      tokenized_annotation = transform_caption(annotation, self.vocab, self.max_token_length)

      image_path = os.path.join(self.root, image_name)
      img = Image.open(image_path).convert("RGB")

      if self.transform:
            img = self.transform(img)

      return img, tokenized_annotation
   
   def __len__(self):
      return len(self.image_path)