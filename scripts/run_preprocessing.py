"""
Data Preprocessing Script for HumAID-all Dataset
================================================
Fetches dataset from HuggingFace and applies preprocessing.
"""

import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import os

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize resources
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

print("=" * 70)
print("DATA PREPROCESSING - HumAID-all Dataset")
print("=" * 70)

# ============================================================================
# Step 1: Load Dataset
# ============================================================================
print("\n[Step 1] Loading HumAID-all dataset from HuggingFace...")

from datasets import load_dataset

# Load in streaming mode to avoid split verification issues
dataset = load_dataset("QCRI/HumAID-all", streaming=True)
print(f"Available splits: {list(dataset.keys())}")

# Collect all train data
print("\nCollecting train split data...")
train_data = list(dataset['train'])
print(f"Loaded {len(train_data)} training examples")

# Convert to DataFrame
df = pd.DataFrame(train_data)
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSample data:")
print(df.head(3))

# ============================================================================
# Step 2: Define Preprocessing Functions
# ============================================================================
print("\n[Step 2] Preprocessing functions defined")

def remove_urls(text):
    if not isinstance(text, str):
        return ""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)

def remove_mentions(text):
    if not isinstance(text, str):
        return ""
    mention_pattern = r'@\w+'
    return re.sub(mention_pattern, '', text)

def remove_hashtags(text):
    if not isinstance(text, str):
        return ""
    hashtag_pattern = r'#(\w+)'
    return re.sub(hashtag_pattern, r'\1', text)

def remove_numbers(text):
    if not isinstance(text, str):
        return ""
    number_pattern = r'\b\d+\b'
    return re.sub(number_pattern, '', text)

def remove_special_chars(text):
    if not isinstance(text, str):
        return ""
    special_chars = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = special_chars.sub('', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def clean_text(text, remove_stop=True, do_lemmatize=True):
    if not isinstance(text, str) or not text:
        return ""

    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_numbers(text)
    text = remove_special_chars(text)
    text = text.lower()
    tokens = word_tokenize(text)

    if remove_stop:
        tokens = [token for token in tokens if token not in STOPWORDS]

    if do_lemmatize:
        tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    cleaned = ' '.join(tokens)
    cleaned = ' '.join(cleaned.split())
    return cleaned

# ============================================================================
# Step 3: Test Preprocessing
# ============================================================================
print("\n[Step 3] Testing preprocessing on sample data...")

test_cases = [
    "HELP!!! http://t.co/xx Building on fire @john #disaster 123",
    "Earthquake hit Nepal, 5000 injured #earthquake @news",
    "SOS! People trapped in flood water, need help urgently!",
]

for test in test_cases:
    print(f"  Raw:   {test[:50]}...")
    print(f"  Clean: {clean_text(test)}")

# ============================================================================
# Step 4: Apply Preprocessing to Dataset
# ============================================================================
print(f"\n[Step 4] Preprocessing {len(df)} texts...")

# Identify text column
text_column = 'tweet_text' if 'tweet_text' in df.columns else df.columns[0]
print(f"Using text column: '{text_column}'")

df['clean_text'] = df[text_column].apply(clean_text)
print("Preprocessing complete!")

# Add length features
df['original_length'] = df[text_column].apply(lambda x: len(str(x)))
df['cleaned_length'] = df['clean_text'].apply(lambda x: len(x))
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

print(f"\nText length statistics:")
print(f"  Original - Mean: {df['original_length'].mean():.1f}, Min: {df['original_length'].min()}, Max: {df['original_length'].max()}")
print(f"  Cleaned  - Mean: {df['cleaned_length'].mean():.1f}, Min: {df['cleaned_length'].min()}, Max: {df['cleaned_length'].max()}")
print(f"  Word count - Mean: {df['word_count'].mean():.1f}")

# ============================================================================
# Step 5: Generate Visualizations
# ============================================================================
print("\n[Step 5] Generating visualizations...")

# Create output directory
os.makedirs('../reports/figures', exist_ok=True)

# Text length distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df['original_length'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Original Text Length Distribution')
axes[0, 0].set_xlabel('Character Count')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['original_length'].mean(), color='red', linestyle='--')

axes[0, 1].hist(df['cleaned_length'], bins=50, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Cleaned Text Length Distribution')
axes[0, 1].set_xlabel('Character Count')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df['cleaned_length'].mean(), color='red', linestyle='--')

axes[1, 0].hist(df['word_count'], bins=30, color='salmon', edgecolor='black')
axes[1, 0].set_title('Word Count Distribution')
axes[1, 0].set_xlabel('Word Count')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(df['word_count'].mean(), color='red', linestyle='--')

axes[1, 1].scatter(df['original_length'][:500], df['cleaned_length'][:500], alpha=0.5)
axes[1, 1].plot([0, df['original_length'].max()], [0, df['original_length'].max()], 'r--')
axes[1, 1].set_title('Original vs Cleaned Length')
axes[1, 1].set_xlabel('Original')
axes[1, 1].set_ylabel('Cleaned')

plt.tight_layout()
plt.savefig('../reports/figures/text_length_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - Saved: text_length_analysis.png")

# Word Cloud
all_text = ' '.join(df['clean_text'].dropna())
plt.figure(figsize=(15, 10))
wordcloud = WordCloud(width=1600, height=800, background_color='white',
                      max_words=200, colormap='viridis').generate(all_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Cleaned Disaster Tweets')
plt.savefig('../reports/figures/wordcloud_cleaned.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - Saved: wordcloud_cleaned.png")

# Top 20 words
all_words = all_text.split()
word_freq = Counter(all_words)
top_20_words = word_freq.most_common(20)
top_words_df = pd.DataFrame(top_20_words, columns=['Word', 'Frequency'])

plt.figure(figsize=(12, 6))
plt.barh(top_words_df['Word'], top_words_df['Frequency'], color='steelblue')
plt.gca().invert_yaxis()
plt.title('Top 20 Most Frequent Words')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.tight_layout()
plt.savefig('../reports/figures/top_20_words.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - Saved: top_20_words.png")

# Class distribution
if 'class_label' in df.columns:
    plt.figure(figsize=(10, 6))
    class_counts = df['class_label'].value_counts()
    class_counts.plot(kind='bar', color='steelblue')
    plt.title('Class Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../reports/figures/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  - Saved: class_distribution.png")

# ============================================================================
# Step 6: Save Preprocessed Data
# ============================================================================
print("\n[Step 6] Saving preprocessed data...")

os.makedirs('../data/processed', exist_ok=True)
output_path = '../data/processed/humaid_preprocessed.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"Saved: {output_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)
print(f"\nDataset Info:")
print(f"  Total rows: {len(df)}")
print(f"  Columns: {df.columns.tolist()}")
print(f"\nOutput Files:")
print(f"  - ../data/processed/humaid_preprocessed.csv")
print(f"  - ../reports/figures/text_length_analysis.png")
print(f"  - ../reports/figures/wordcloud_cleaned.png")
print(f"  - ../reports/figures/top_20_words.png")
if 'class_label' in df.columns:
    print(f"  - ../reports/figures/class_distribution.png")

print(f"\nSample preprocessed data:")
print(df[['tweet_text', 'clean_text', 'class_label']].head(5).to_string())

print("\n" + "=" * 70)
print("Next: Notebook 03 - Disaster Classification")
print("=" * 70)
