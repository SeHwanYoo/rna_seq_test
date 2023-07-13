from PIL import Image
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, ViTFeatureExtractor, AutoFeatureExtractor
# from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator
from transformers import AdamW


import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor,
                                    ToPILImage)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_data(filename, pkl_file):
    with open(filename, 'wb') as f:
	    pickle.dump(pkl_file, f, protocol=pickle.HIGHEST_PROTOCOL)

def find_key(json_obj, target_value):
    for ii in range(len(json_obj)):
        for key, value in json_obj[str(ii)].items():
            if value == target_value:
                return json_obj[str(ii)]['caption']
    return None


class CancerDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, tokenizer, feature_extractor, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.tokenizer= tokenizer
        self.feature_extractor = feature_extractor
        # self.max_length = 2129
        self.max_target_length = 600
        self.images_size = feature_extractor.size["height"]

    def __getitem__(self, idx):
        lbl = self.labels[idx]
        img = Image.open(self.images[idx])

        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((self.images_size, self.images_size))

        img = self.feature_extractor(img, return_tensors="pt").pixel_values
        item = {'images': torch.tensor(img.squeeze()), 'labels' : torch.tensor(self._tokenization(lbl))}

        return item

    def __len__(self):
        return len(self.images)

    def _tokenization(self, captions):
        labels = tokenizer(captions, padding="max_length", max_length=self.max_target_length).input_ids
        return labels
    
    
def train(model, optim, train_dataset):
    model.train()

    # optim = AdamW(model.parameters(), lr=5e-5)
    optim = AdamW(model.parameters(), lr=1e-5)
    train_loss_list = []
    for epoch in range(100):
        train_loss = 0
        for batch in train_dataset:

            labels = batch['labels'].to(device)
            images = batch['images'].to(device)

            outputs = model(pixel_values=images, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

            train_loss += loss.item()

        print(epoch, '/100 Loss : ', train_loss)
        train_loss_list.append(train_loss)
        
    save_data('train_loss.pickle', train_loss)
    torch.save(model.state_dict(), f'model_weights_am.pth')


if __name__ == '__main__':
    path = 'D:/Dropbox/University/2. NTU/Ph.D project/books_set'
    
    # images 
    cancer_images = glob(os.path.join(path, '*/*.png'))
    
    # json 
    with open(os.path.join(path, 'captions.json'), 'r') as f:
        captions = json.load(f)
        
    # print(captions)
        
    cancer_labels = []
    # max_cap = 0
    for img in cancer_images:
        uuid = img.split('\\')[-1].split('.')[0]
        cap = find_key(captions, uuid)

        # if len(cap) > max_cap:
        #     max_cap = len(cap)

        cancer_labels.append(cap)
        
    # print(cancer_images[:2])
    # print(cancer_labels[:2])
    # exit() 
    
    image_encoder_model = "google/vit-base-patch16-224-in21k"
    text_decode_model = "gpt2"

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(image_encoder_model, text_decode_model)
    model.to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
    
    image_mean, image_std = feature_extractor.image_mean, feature_extractor.image_std
    size = feature_extractor.size["height"]

    _transforms = Compose(
            [
                # RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=image_mean, std=image_std),
                # ToPILImage()
            ]
        )
    
    
    tokenizer = AutoTokenizer.from_pretrained(text_decode_model)

    tokenizer.pad_token = tokenizer.eos_token

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    optim = AdamW(model.parameters(), lr=1e-5)
    
    Train_dataset = CancerDataset(cancer_images, cancer_labels, tokenizer, feature_extractor, transform=_transforms)
    train_dataset = DataLoader(Train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Test_dataset = CancerDataset(test_images, test_labels, tokenizer, feature_extractor, transform=_transforms)
    # test_dataset = DataLoader(Test_dataset, batch_size=4, shuffle=False)
    
    train(model, optim, Train_dataset)