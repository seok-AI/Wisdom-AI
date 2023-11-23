import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import random
import warnings
import torch.nn.functional as F
import timm
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


warnings.filterwarnings(action='ignore')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())
## Hyperparameter Settings
# print("Available Vision Transformer Models: ")
# timm.list_models("swinv2*")
CFG = {
        'IMG_SIZE':256,
        'EPOCHS':30, #Your Epochs,
        'LR':1e-4, #Your Learning Rate,
        'BATCH_SIZE':8, #Your Batch Size,
        'SEED':41
        }
## Fixed Random-Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed | | ~U

## Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filename']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # mos column 존~^ ~W~@~W~P ~T~] ~R~]~D ~D| ~U
        mos = float(self.dataframe.iloc[idx]['time_min']) if 'time_min' in self.dataframe.columns else 0.0
        return img, mos

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filename']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        table = self.dataframe.iloc[idx]
        if 'time_min' in list(table.index):
            table = table.drop(['filename','time_min'])
        else:
            table = table.drop(['filename'])


        table = torch.tensor(list(table))

        # mos column 존~^ ~W~@~W~P ~T~] ~R~]~D ~D| ~U
        mos = float(self.dataframe.iloc[idx]['time_min']) if 'time_min' in self.dataframe.columns else 0.0
        return img, table, mos


def MLP_Module(in_dim, out_dim):
    return nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
            )


 ## Define Model
class CNN_ViT(nn.Module):
    def __init__(self):
        super(CNN_ViT, self).__init__()

        self.backbone = timm.create_model("swinv2_large_window12to16_192to256", pretrained=False)
        self.backbone.load_state_dict(torch.load('/USER/huggingface/hub/swinv2_large_window12to16_192to256.pt'))
        self.backbone.head.fc = nn.Identity()
        self.table_backbone = nn.Sequential(
                MLP_Module(45,128),
                MLP_Module(128,512),
                MLP_Module(512,2048),
                MLP_Module(2048,2048),
                MLP_Module(2048,2048),
                MLP_Module(2048,1536),
                )

        # Image quality assessment head
        self.regression_head = nn.Linear(1536*2, 1)

    def forward(self, images, table):
        # Swin V2
        features_vit = self.backbone(images)
        features_table = self.table_backbone(table)

        #features = features_vit + features_table
        features = torch.cat([features_vit, features_table], axis=1)
        # Image quality regression
        mos = self.regression_head(features)

        return mos



path = '/DATA/'

train_data = pd.read_csv(path + 'train/train.csv')
test_data = pd.read_csv(path + 'test/test.csv')

def rewrite(df, train=True):
    if train == False:
        df['filename'] = path + 'test/images/' + df['filename']
    else:
        df['filename'] = path + 'train/images/' + df['filename']
    return df

train_data = rewrite(train_data)
test_data = rewrite(test_data, train=False)
train_data, val_data = train_test_split(train_data, test_size = 0.08, shuffle=True, random_state=41)

train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ordinal_features = ['operator', 'sex']

for feature in ordinal_features:
    le = LabelEncoder()
    le = le.fit(train_data[feature])
    train_data[feature] = le.transform(train_data[feature])
    for label in np.unique(val_data[feature]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    val_data[feature] = le.transform(val_data[feature])
    for label in np.unique(test_data[feature]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test_data[feature] = le.transform(test_data[feature])


have_nan = ['height', 'weight', 'bmi']
for col in have_nan:
    train_data.loc[train_data[col].isna(), col] = train_data[col].mean()

for col in have_nan:
    val_data.loc[val_data[col].isna(), col] = train_data[col].mean()

for col in have_nan:
    test_data.loc[test_data[col].isna(), col] = train_data[col].mean()


from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PolynomialFeatures

scaler = PolynomialFeatures(degree=2)
std_train = pd.DataFrame(scaler.fit_transform(train_data.drop(columns=['time_min','ID','filename','date'])))
std_val = pd.DataFrame(scaler.transform(val_data.drop(columns=['time_min','ID','filename','date'])))
std_test = pd.DataFrame(scaler.transform(test_data.drop(columns=['ID','filename','date'])))

std_train['time_min'] = train_data['time_min']
std_val['time_min'] = val_data['time_min']

std_train['filename'] = train_data['filename']
std_val['filename'] = val_data['filename']
std_test['filename'] = test_data['filename']

train_data = std_train.copy()
val_data = std_val.copy()
test_data = std_test.copy()

scaler = StandardScaler()
std_train = pd.DataFrame(scaler.fit_transform(train_data.drop(columns=['time_min','filename'])))
std_val = pd.DataFrame(scaler.transform(val_data.drop(columns=['time_min','filename'])))
std_test = pd.DataFrame(scaler.transform(test_data.drop(columns=['filename'])))

std_train['time_min'] = train_data['time_min']
std_val['time_min'] = val_data['time_min']

std_train['filename'] = train_data['filename']
std_val['filename'] = val_data['filename']
std_test['filename'] = test_data['filename']

train_data = std_train.copy()
val_data = std_val.copy()
test_data = std_test.copy()





def test(model, test_loader):
    model.eval()
    predicted_mos_list = []
    test_loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for imgs, table, mos in test_loop:
            imgs, table = imgs.float().to(device), table.float().to(device)

            predicted_mos = model(imgs, table)
            predicted_mos_list += predicted_mos.reshape(-1).tolist()

    return predicted_mos_list


## Train
def val(model, criterion1, val_loader, epoch):
    model.eval()
    val_loss = 0
    predicted_mos_list, real = [], []
    val_loop = tqdm(val_loader, leave=True)
    with torch.no_grad():
        for imgs, table, mos in val_loop:
            imgs, table, mos = imgs.float().to(device), table.float().to(device), mos.float().to(device)
            # Forward & Loss
            predicted_mos = model(imgs, table)
            loss = criterion1(predicted_mos.squeeze(1), mos)
            predicted_mos_list += predicted_mos.reshape(-1).tolist()
            real += mos.reshape(-1).tolist()

            val_loss += loss.item()
            val_loop.set_description(f"Validation")
            val_loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1} val average loss: {val_loss / len(val_loader):.4f}, mape: {mean_absolute_percentage_error(real, predicted_mos_list):.4f}\n")
def train(model, criterion1, optimizer, train_loader, val_loader, scheduler=None):
# ~U~Y~J
    for epoch in range(CFG['EPOCHS']):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=True)
        predicted_mos_list, real = [], []
        for imgs, table, mos in loop:
            imgs, table, mos = imgs.float().to(device), table.float().to(device), mos.float().to(device)
            # Forward & Loss
            predicted_mos = model(imgs, table)
            # loss0 = criterion0(predicted_mos, mos)
            loss = criterion1(predicted_mos.squeeze(1), mos)

            predicted_mos_list += predicted_mos.reshape(-1).tolist()
            real += mos.reshape(-1).tolist()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Train")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} finished with average loss: {total_loss / len(train_loader):.4f}, mape: {mean_absolute_percentage_error(real, predicted_mos_list):.4f}")
        val(model, criterion1, val_loader, epoch)
        if scheduler != None:
            scheduler.step()

transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])




# 모~M, ~F~P~K~U~H~X, ~X~K~H~]| ~@
model = CNN_ViT()
model.to(device)
print('done.')

#for name, param in model.named_parameters():
#    print(name, param.requires_grad)



criterion = nn.MSELoss()
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG['EPOCHS'], eta_min=1e-7)

all_parameters_except_image_encoder = [param for name, param in model.named_parameters() if "table" not in name]
optimizer = torch.optim.AdamW([
        {'params': model.table_backbone.parameters(), 'lr': 1e-3},
            {'params': all_parameters_except_image_encoder, 'lr': 1e-5}
            ])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG['EPOCHS'], eta_min=1e-7)


train_dataset = CustomDataset(train_data, transform)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4)

val_dataset = CustomDataset(val_data, transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4)


test_dataset = CustomDataset(test_data, transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4)

train(model, criterion, optimizer, train_loader, val_loader)

prediction = test(model, test_loader)


sample_submission = pd.read_csv('/DATA/test/sample_submission.csv')

sample_submission['time_min'] = prediction

sample_submission.to_csv('/USER/result/PolynomialFeatures.csv')
print(sample_submission)


