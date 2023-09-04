import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from torchvision.transforms import ToTensor

file = open('captions.txt','r')

file_content = file.readlines()

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


class CaptionDataset(torch.utils.data.Dataset):
   def __init__(self, caption_array, image_path, root, transform=None): #Permet d'initialiser les variables avant d'accéder a un élément précis du dataset

      self.caption_array = caption_array
      self.root = root
      self.transform = transform
      self.image_path = image_path

   def __getitem__(self, index): #Permet d'accéder à un élément spécifique du dataset
      annotation = self.caption_array[index]
      image_name = self.image_path[index]

      image_path = os.path.join(self.root, image_name)

      img = Image.open(image_path).convert("RGB")

      if self.transform:
            img = self.transform(img)

      return img, annotation
   
   def __len__(self):
      return len(self.image_path)
   

root = "C:\\Users\\Audensiel\\Desktop\\piccture_anotation\\Images"   
dataset = CaptionDataset(image_annot, image_path, root)
sample_index = 1 
sample_img, sample_caption = dataset[sample_index]
print("Sample Caption:", sample_caption)
print("Sample Image Shape:", sample_img.size)
sample_img.show()