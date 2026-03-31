"""
Text Preprocessing Module for Disaster Response AI System
==========================================================
Cleans and preprocesses raw social media text for ML models.

Functions:
    - clean_text: Main preprocessing pipeline
    - remove_urls: Remove HTTP/URL links
    - remove_mentions: Remove @user mentions
    - remove_hashtags: Remove # but keep the word
    - remove_numbers: Remove standalone numbers
    - remove_special_chars: Remove emojis and symbols
    - tokenize: Split text into tokens
    - remove_stopwords: Remove common English stopwords
    - lemmatize: Reduce words to base form
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Optional

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK resources."""
    resources = [
        'stopwords',
        'punkt',
        'punkt_tab',
        'wordnet',
        'omw-1.4'
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

# Initialize resources
download_nltk_data()
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)


def remove_mentions(text: str) -> str:
    """Remove @mentions from text."""
    mention_pattern = r'@\w+'
    return re.sub(mention_pattern, '', text)


def remove_hashtags(text: str) -> str:
    """Remove # symbol but keep the hashtag word."""
    hashtag_pattern = r'#(\w+)'
    return re.sub(hashtag_pattern, r'\1', text)


def remove_numbers(text: str) -> str:
    """Remove standalone numbers from text."""
    number_pattern = r'\b\d+\b'
    return re.sub(number_pattern, '', text)


def remove_special_chars(text: str) -> str:
    """Remove special characters and emojis."""
    # Remove emojis and special unicode characters
    special_chars = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = special_chars.sub('', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove English stopwords from token list."""
    return [token for token in tokens if token.lower() not in STOPWORDS]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens to their base form."""
    return [LEMMATIZER.lemmatize(token) for token in tokens]


def clean_text(
    text: str,
    remove_stop: bool = True,
    do_lemmatize: bool = True,
    return_tokens: bool = False
) -> str:
    """
    Main text cleaning pipeline.

    Args:
        text: Raw input text
        remove_stop: Whether to remove stopwords (default: True)
        do_lemmatize: Whether to lemmatize (default: True)
        return_tokens: Return tokenized output (default: False)

    Returns:
        Cleaned text string or list of tokens

    Example:
        >>> raw = "HELP!!! http://t.co/xx Building on fire @john #disaster 123"
        >>> clean_text(raw)
        'help building fire disaster'
    """
    if not isinstance(text, str) or not text:
        return ""

    # Step 1: Remove URLs
    text = remove_urls(text)

    # Step 2: Remove mentions
    text = remove_mentions(text)

    # Step 3: Remove # but keep hashtag words
    text = remove_hashtags(text)

    # Step 4: Remove numbers
    text = remove_numbers(text)

    # Step 5: Remove special characters and emojis
    text = remove_special_chars(text)

    # Step 6: Lowercase
    text = text.lower()

    # Step 7: Tokenize
    tokens = tokenize(text)

    # Step 8: Remove stopwords
    if remove_stop:
        tokens = remove_stopwords(tokens)

    # Step 9: Lemmatize
    if do_lemmatize:
        tokens = lemmatize_tokens(tokens)

    # Remove extra whitespace and return
    cleaned = ' '.join(tokens)
    cleaned = ' '.join(cleaned.split())  # Remove multiple spaces

    if return_tokens:
        return cleaned.split()
    return cleaned


def preprocess_dataframe(df, text_column='text', clean_column='clean_text'):
    """
    Apply preprocessing to a pandas DataFrame.

    Args:
        df: Input DataFrame
        text_column: Name of column containing raw text
        clean_column: Name of column for cleaned text

    Returns:
        DataFrame with added clean_text column
    """
    df[clean_column] = df[text_column].apply(clean_text)
    return df


if __name__ == "__main__":
    # Test examples
    test_cases = [
        "HELP!!! http://t.co/xx Building on fire @john #disaster 123",
        "Earthquake hit Nepal, 5000 injured #earthquake @news",
        "Just watching a movie, nothing special",
        "SOS! People trapped in flood water, need help urgently!",
        "",
        None,
        "🔥🔥 FIRE at mall, everyone run! 🆘"
    ]

    print("=" * 60)
    print("TEXT PREPROCESSING TEST CASES")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        if test is None:
            print(f"  Raw:     None")
        else:
            # Encode to avoid Windows console unicode issues
            raw_ascii = test.encode('ascii', errors='replace')[:60]
            print(f"  Raw:     {raw_ascii}")
        cleaned = clean_text(test)
        print(f"  Cleaned: {cleaned}")

    print("\n" + "=" * 60)
