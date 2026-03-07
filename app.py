
import streamlit as st
st.set_page_config(page_title="Hindi Meme Sentiment", page_icon="📸", layout="centered", initial_sidebar_state="auto")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from transformers import CLIPProcessor, CLIPModel
import pytesseract  
import numpy as np

class FusionModel(nn.Module):
    def __init__(self, img_dim=512, text_dim=1024, num_classes=3):
        super(FusionModel, self).__init__()
        self.fusion_layer = nn.Linear(img_dim + text_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, img_feat, text_feat):
        x = torch.cat((img_feat, text_feat), dim=1)
        x = self.fusion_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"

model = FusionModel(img_dim=512, text_dim=1024, num_classes=3).to(device)
state_dict = torch.load("fusion\\best_fusion_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

text_model_name = "google_muril/best_hindi_muril_large"
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_encoder = AutoModel.from_pretrained(text_model_name, low_cpu_mem_usage=False).to(device)
text_encoder.eval()

clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name, low_cpu_mem_usage=False).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model.eval()

st.title("📸 Hindi Meme Sentiment Analysis")
st.write("Upload a meme image → get sentiment prediction directly.")

uploaded_img = st.file_uploader("Upload Meme Image", type=["jpg", "png", "jpeg"])

if st.button("Predict"):
    if uploaded_img:
        # 1. Load image
        image = Image.open(uploaded_img).convert("RGB")
        
        # 2. Extract caption using OCR (if any)
        caption_text = pytesseract.image_to_string(image, lang='hin').strip()

        # 3. Image embedding
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feat = clip_model.get_image_features(**inputs)
        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)

        # 4. Text embedding (if OCR found text)
        if caption_text:
            tokens = tokenizer(
                caption_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            ).to(device)
            with torch.no_grad():
                text_outputs = text_encoder(**tokens)
            text_feat = text_outputs.last_hidden_state[:, 0, :]  # CLS
            text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
        else:
            text_feat = torch.zeros((1, 1024)).to(device)

        # 5. Fusion prediction
        with torch.no_grad():
            logits = model(img_feat, text_feat)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # 6. Display result
        labels = ["negative", "neutral", "positive"]
        st.write("### 🔮 Prediction:")
        for i, label in enumerate(labels):
            st.write(f"{label.capitalize()}: {probs[i]*100:.2f}%")
        
        # Find the index of the sentiment with the highest percentage
        max_prob_index = np.argmax(probs)
        
        # Get the name of the winning sentiment
        final_sentiment = labels[max_prob_index]
        
        # Get the percentage value of the winning sentiment
        final_percentage = probs[max_prob_index]
        
        st.write("### 🔮 Final Answer:")
        st.subheader(f"The predicted sentiment is **{final_sentiment.capitalize()}** ")
    else:
        st.warning("Please upload an image!")
