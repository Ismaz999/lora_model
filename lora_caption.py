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

from flickr8k_manager import CaptionDataset

file = open('captions.txt','r')
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

transform_img = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])


root = "C:\\Users\\Audensiel\\Desktop\\piccture_anotation\\Images"   
dataset = CaptionDataset(image_annot, image_path, root, transform=transform_img)


resnet_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
model_features = torch.nn.Sequential(*list(resnet50.children())[:-2])
model_features.eval()

test_resnet = Subset(dataset, indices=list(range(50)))
input_data = DataLoader(test_resnet, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_features.to(device)

features = []

with torch.no_grad():
   for batch_images, batch_captions in input_data:
      batch_images = batch_images.to(device)
      batch_outputs = model_features(batch_images)

      all_outputs = features.extend(batch_outputs)

   for i in range(batch_images.shape[0]):
      image = batch_images[i].cpu().permute(1, 2, 0)
      output = batch_outputs[i] 
            
      plt.figure()
      plt.imshow(image)
      plt.show()