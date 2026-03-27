# 🚨 AI-Based Disaster Management & Emergency Response System

An AI-powered **Multimodal** disaster detection system that analyzes
social media **text, images, and videos** to identify real disaster
events, determine urgency levels, detect fake posts, and pinpoint
geographic locations to alert emergency responders instantly.

---

## 🎯 How It Works
```
Social Media Post (Text + Image + Video)
              ↓
    AI Analysis System
              ↓
  ✅ Real Disaster Confirmed
  🔴 Urgency: CRITICAL
  📍 Location: Detected & Mapped
  🚒 Alert sent to Responders!
```

---

## 📦 System Modules

| Module | Description | Status |
|--------|-------------|--------|
| 📥 Data Collection | Gathers disaster datasets | In Progress  |
| 🧹 Text Preprocessing | Cleans tweet text |  |
| 🤖 Disaster Classifier | Text-based disaster detection |  |
| ⚡ Urgency Classifier | Rates urgency level |  |
| 🚫 Misinformation Filter | Detects fake posts |  |
| 📸 Image Analysis | CNN-based image classifier |  |
| 🎥 Video Analysis | Frame-by-frame video analysis | |
| 📍 Location Detection | GPS & text-based geo location |  |
| 📊 Dashboard | Live map + charts + alerts | |

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| NLP | NLTK, spaCy |
| ML Models | Scikit-learn |
| Image/Video | OpenCV, TensorFlow/Keras |
| Location | Geopy, Folium |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Plotly |

---

## 📁 Project Structure
```
disaster-response-ai/
├── data/
│   ├── raw/          → Text datasets
│   ├── processed/    → Cleaned data
│   ├── images/       → Disaster images
│   │   ├── disaster/
│   │   └── non_disaster/
│   └── videos/       → Disaster videos
├── notebooks/        → Jupyter notebooks
├── src/
│   ├── data_collection/
│   ├── preprocessing/ → Text + Image + Video
│   ├── models/        → All AI models
│   ├── location/      → GPS detection
│   └── utils/
├── dashboard/        → Streamlit app
├── scripts/
├── models/           → Saved model files
└── reports/
```

---

## 🚀 Setup
```bash
git clone https://github.com/Nandan-BV/disaster-response-ai.git
cd disaster-response-ai
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## 📊 Notebooks

| # | Notebook | Status |
|---|----------|--------|
| 1 | Data Exploration | In Progress |
| 2 | Text Preprocessing | |
| 3 | Disaster Classification | |
| 4 | Urgency Classification | |
| 5 | Misinformation Filter |  |
| 6 | Image Analysis | |
| 7 | Video Analysis | |
| 8 | Location Detection | |

---

## 👥 Team
- Nandan— Developer
- Tilak D R - AI Developer
- Srushti Joshi - AI/ML Developer
- P Suhas - BodyBuilder
