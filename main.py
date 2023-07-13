import json
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from transformers import AutoTokenizer,AutoModel
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with open('train.json', 'r') as f:
    train_data = json.load(f)
with open('val.json', 'r') as f:
    val_data = json.load(f)
with open('test.json', 'r') as f:
    test_data = json.load(f)
# print(train_data[2])

# 加载预训练模型
tokenizer = AutoTokenizer.from_pretrained("./token")
emb_token = AutoModel.from_pretrained('./token').to(device)

# 处理图片，缩放到统一大小
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 提取文字、图片信息

def extract(dataset):
    for i in range(len(dataset)):
        # for i in range(3):
        temp = dataset[i]['text']
        encoded_input = tokenizer(temp, padding=True, truncation=True, max_length=50, return_tensors='pt')
        dataset[i]['input_ids'] = encoded_input['input_ids'].squeeze(1)
        input_ids = encoded_input['input_ids'].squeeze()
        dataset[i]['input_ids'] = torch.nn.functional.pad(input_ids, pad=(0, 50 - len(input_ids)), mode='constant', value=0)
        attention_mask = torch.zeros(50, dtype=torch.long)
        attention_mask[:len(input_ids)] = 1
        dataset[i]['attention_mask'] = attention_mask
        #dataset[i]['text'] = emb_token(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask']).last_hidden_state.squeeze(dim=0)
        # print(dataset[i]['text'])
        photo_path = dataset[i]['photo']
        photo_info = Image.open(photo_path)
        photo_info = transform(photo_info)
        dataset[i]['photo'] = photo_info
        # print(dataset[i]['photo'])
    return dataset


ex_train = extract(train_data)
ex_val = extract(val_data)
ex_test = extract(test_data)


# print(ex_val[1])


# 文本特征提取模型定义
class TextModle(nn.Module):
    def __init__(self):
        super(TextModle, self).__init__()
        self.textcnn = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, input_ids):
        input_ids = input_ids.permute(0, 2, 1)
        output = self.textcnn(input_ids)
        output = F.adaptive_max_pool1d(output, 1).squeeze(2)  # 池化后的输出形状为[B, C, 1]，压缩维度成[B, C]
        return output


# 图片特征提取模型定义
class PhotoModel(nn.Module):
    def __init__(self):
        super(PhotoModel, self).__init__()
        self.resnet = resnet50(pretrained=True)  # 使用预训练的ResNet-50作为图片特征提取器

    def forward(self, image):
        features = self.resnet(image)
        return features


class MutiModel(nn.Module):
    def __init__(self, num_classes, choice):
        super(MutiModel, self).__init__()
        self.image_extractor = PhotoModel()
        self.text_encoder = TextModle()
        self.choice = choice
        # 仅输入图像特征
        self.classifier0 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
        )
        # 仅输入文本特征
        self.classifier1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
        )
        # 多模态融合
        self.classifier2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1064, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, image, text):
        if self.choice == 0:
            image_features = self.image_extractor(image)
            output = self.classifier0(image_features)
        elif self.choice == 1:
            text_features = self.text_encoder(text)
            output = self.classifier1(text_features)
        else:
            image_features = self.image_extractor(image)
            text_features = self.text_encoder(text)
            Muti_features = torch.cat((text_features, image_features), dim=-1)
            output = self.classifier2(Muti_features)
        return output


# 训练过程
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    total_correct = 0
    for guid, labels, images, input_ids, attention_mask in train_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text = emb_token(input_ids, attention_mask=attention_mask).last_hidden_state
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, text)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = total_correct.item() / len(train_loader.dataset)
    return epoch_loss, epoch_acc


# 预测过程
def predict_model(model, test_loader, device):
    model.eval()
    total_correct = 0
    for guid, labels, images, input_ids, attention_mask in test_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text = emb_token(input_ids, attention_mask=attention_mask).last_hidden_state
        with torch.no_grad():
            outputs = model(images, text)
            _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels.to(device))
    return total_correct.item() / len(test_loader.dataset)


# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--choice', default=2, help='选0仅输入图片信息,选1仅输入文本信息,默认选2输入融合信息',type=int)
parser.add_argument('--lr', default=5e-5, help='设置学习率', type=float)
args = parser.parse_args()
choice = args.choice
lr = args.lr

epoch = 10
batch_size = 16
best_acc = 0
criterion = nn.CrossEntropyLoss()

def collate_fn(data_list):
    guid = []
    label = []
    input_ids = []
    attention_mask = []
    photo = []
    for data in data_list:
        guid += [data['guid']]
        label += [data['label']]
        input_ids += [data['input_ids']]
        attention_mask += [data['attention_mask']]
        photo += [data['photo']]
    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    photo = torch.stack(photo, dim=0)
    return guid, torch.tensor(label), photo, input_ids, attention_mask


# 构建train和test的DataLoader
loader_train = DataLoader(ex_train, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(ex_val, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)


# 模型训练
model = MutiModel(3, choice)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epoch):
    train_loss, train_acc = train_model(model, loader_train, criterion, optimizer, device)
    val_acc = predict_model(model, loader_val, device)
    if (val_acc > best_acc):
        best_acc = val_acc
        torch.save(model, 'multi_model.pt')
    print(
        f"Epoch {epoch + 1}/{epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc:{best_acc:.4f}")

loader_test = DataLoader(ex_test, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

#用正确率最高的模型进行预测
best_model = torch.load('multi_model.pt').to(device)
best_model.eval()
guid_all = []
test_predictions = []
for guid, labels, images, input_ids, attention_mask in loader_test:
    images = images.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    text = emb_token(input_ids, attention_mask=attention_mask).last_hidden_state
    with torch.no_grad():
        outputs = best_model(images, text)
        _, preds = torch.max(outputs, 1)
        guid_all += guid
        test_predictions += preds

#生成预测文件
guid_all = torch.tensor(guid_all).squeeze()
test_predictions = torch.tensor(test_predictions).squeeze()
test_df = pd.DataFrame(data = {'guid':guid_all,'label':test_predictions})
column_dict_ = {0:"positive", 1:"negative",2:"neutral"}
test_df['label'] = test_predictions
pre_df = test_df.replace({"label": column_dict_})
pre_df.to_csv('predict.txt',sep=',',index=False)
