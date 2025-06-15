import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from transformers import WhisperProcessor, WhisperModel
import random
from tqdm import tqdm

MODEL_NAME = "./Model_whisper/"
TEST_CSV_PATH = "./data_asr/test.csv"
TEST_AUDIO_DIR = "./data_asr/speechs/speechs/test/"
NUM_LABELS = 6
SAMPLING_RATE = 16000
THRESHOLD = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
whisper_model = WhisperModel.from_pretrained(MODEL_NAME)

# Model WhisperClassifier
class WhisperClassifier(nn.Module):
    def __init__(self, whisper_model=whisper_model, num_labels=NUM_LABELS):
        super().__init__()
        self.encoder = whisper_model.encoder
        self.encoder_block = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=1280, nhead=8, dropout=0.1, batch_first=True, activation='gelu'),
                        num_layers=2
                        )
        self.classifier = nn.Sequential(
            nn.Linear(1280, num_labels),
        )
        self.weight_proj = nn.Linear(1280, 1)

    def forward(self, input_features_1, input_features_2):
        outputs_1 = self.encoder(input_features=input_features_1).last_hidden_state
        outputs_2 = self.encoder(input_features=input_features_2).last_hidden_state
        cat_outputs = torch.cat([outputs_1, outputs_2], dim=1)
        x_attn = self.encoder_block(cat_outputs)
        weights = torch.softmax(self.weight_proj(x_attn), dim=1)
        pooled = (x_attn * weights).sum(dim=1)
        logits = self.classifier(pooled)
        return logits


# Data loader for Test Data
class WhisperTestDataset(Dataset):
    def __init__(self, csv_path, audio_dir, processor, sampling_rate=16000):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['id'] + '.wav')
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)(waveform)
        waveform = waveform.squeeze(0)
        chunck_lst = []
        for num_chuck in range(2):
            inputs = self.processor(waveform[num_chuck * 480000:(num_chuck+1) * 480000], sampling_rate=self.sampling_rate, return_tensors="pt")
            input_features = inputs.input_features.squeeze(0)
            chunck_lst.append(input_features)
        #inputs = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt")
        #input_features = inputs.input_features.squeeze(0)
        return row['id'], {"input_features_1": chunck_lst[0], "input_features_2": chunck_lst[1]}


# Define Model
model = WhisperClassifier(num_labels=NUM_LABELS).to(DEVICE)
# Load checkpoint
model.load_state_dict(torch.load("best_model_2trans.pt", weights_only=True))

# Test Loader
test_dataset = WhisperTestDataset(TEST_CSV_PATH, TEST_AUDIO_DIR, processor, sampling_rate=SAMPLING_RATE)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Inference
model.eval()
results = []

with torch.no_grad():
    for audio_id, input_features in tqdm(test_loader):
        input_features_1 = input_features['input_features_1'].to(DEVICE)
        input_features_2 = input_features['input_features_2'].to(DEVICE)
        logits = model(input_features_1, input_features_2)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > THRESHOLD).astype(bool).tolist()[0]
        results.append([audio_id[0]] + preds)

# Save to CSV
label_columns = [
    'กล่าวสวัสดี',
    'แนะนำชื่อและนามสกุล',
    'บอกประเภทใบอนุญาตและเลขที่ใบอนุญาตที่ยังไม่หมดอายุ',
    'บอกวัตถุประสงค์ของการเข้าพบครั้งนี้',
    'เน้นประโยชน์ว่าลูกค้าได้ประโยชน์อะไรจากการเข้าพบครั้งนี้',
    'บอกระยะเวลาที่ใช้ในการเข้าพบ'
]

columns = ['id'] + label_columns
submission_df = pd.DataFrame(results, columns=columns)