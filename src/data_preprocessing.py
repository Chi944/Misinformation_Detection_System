"""
Data acquisition, cleaning, and preprocessing module.
Handles downloading datasets, HTML parsing, deduplication, and temporal stratified splitting.
"""

import re
import hashlib
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import warnings

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED,
    TEST_SIZE, VAL_SIZE, TRAIN_SIZE
)

warnings.filterwarnings('ignore')


class TextPreprocessor:
    """Text preprocessing pipeline for news articles."""
    
    def __init__(self, download_nltk: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            download_nltk: Whether to download required NLTK data
        """
        if download_nltk:
            self._download_nltk_data()
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        packages = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
        for package in packages:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if pd.isna(text):
            return ""
        clean = re.sub(r'<[^>]+>', '', str(text))
        return clean
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and spaces."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        return ' '.join(text.split())
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            return word_tokenize(text.lower())
        except Exception:
            return text.lower().split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        return [t for t in tokens if t not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def preprocess(self, text: str, remove_stops: bool = False, 
                   lemmatize: bool = False) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Input text
            remove_stops: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens
            
        Returns:
            Preprocessed text string
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        text = self.clean_html(text)
        text = self.remove_urls(text)
        text = self.normalize_whitespace(text)
        
        if remove_stops or lemmatize:
            tokens = self.tokenize(text)
            if remove_stops:
                tokens = self.remove_stopwords(tokens)
            if lemmatize:
                tokens = self.lemmatize(tokens)
            text = ' '.join(tokens)
        
        return text
    
    def preprocess_for_bert(self, text: str) -> str:
        """
        Minimal preprocessing for BERT (preserve more structure).
        
        Args:
            text: Input text
            
        Returns:
            Lightly preprocessed text
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        text = self.clean_html(text)
        text = self.remove_urls(text)
        text = self.normalize_whitespace(text)
        
        return text


class DatasetLoader:
    """Load and prepare datasets for misinformation detection."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def create_synthetic_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create a synthetic dataset for testing/demonstration.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic news data
        """
        np.random.seed(RANDOM_SEED)
        
        # Templates for credible news
        credible_templates = [
            "According to official sources, {topic} has been confirmed by multiple experts.",
            "A new study published in {journal} reveals findings about {topic}.",
            "Government officials announced today that {topic} will be implemented.",
            "Researchers at {university} have discovered new evidence regarding {topic}.",
            "The report, verified by independent analysts, shows that {topic}.",
            "Multiple news agencies have confirmed that {topic} occurred yesterday.",
            "Scientists from {institution} presented their findings on {topic} at the conference.",
            "The official statement clarifies the situation regarding {topic}.",
            "Data from reputable sources indicates that {topic} is accurate.",
            "Experts interviewed by major outlets agree that {topic} is significant."
        ]
        
        # Templates for misinformation
        misinfo_templates = [
            "BREAKING: You won't believe what they're hiding about {topic}!!!",
            "SHOCKING REVELATION: The truth about {topic} they don't want you to know!",
            "EXPOSED: Secret plot involving {topic} finally revealed!",
            "URGENT: Share before they delete! The real story about {topic}!",
            "CONSPIRACY CONFIRMED: {topic} is part of a massive cover-up!",
            "Wake up people! {topic} is just the beginning of the deception!",
            "They're lying to you about {topic}! Here's the proof!",
            "BOMBSHELL: Insider reveals the truth about {topic}!",
            "MUST READ: The mainstream media won't tell you this about {topic}!",
            "ALERT: {topic} is being censored! Share now!"
        ]
        
        topics = [
            "the new policy", "vaccine effectiveness", "election results",
            "climate change data", "economic forecasts", "health guidelines",
            "scientific research", "government spending", "international relations",
            "technology developments", "environmental regulations", "public safety measures"
        ]
        
        journals = ["Nature", "Science", "The Lancet", "JAMA", "NEJM"]
        universities = ["Harvard", "MIT", "Stanford", "Oxford", "Cambridge"]
        institutions = ["NIH", "CDC", "WHO", "FDA", "EPA"]
        
        data = []
        
        for i in range(n_samples):
            is_fake = np.random.random() < 0.3  # 30% misinformation
            topic = np.random.choice(topics)
            
            if is_fake:
                template = np.random.choice(misinfo_templates)
                title = template.format(topic=topic)
                body = f"{title} " * np.random.randint(3, 8)
            else:
                template = np.random.choice(credible_templates)
                title = template.format(
                    topic=topic,
                    journal=np.random.choice(journals),
                    university=np.random.choice(universities),
                    institution=np.random.choice(institutions)
                )
                body = f"{title} Additional details support this conclusion. " * np.random.randint(2, 5)
            
            # Generate fake dates for temporal splitting
            year = np.random.choice([2022, 2023, 2024], p=[0.4, 0.4, 0.2])
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            date = f"{year}-{month:02d}-{day:02d}"
            
            data.append({
                'id': i,
                'title': title,
                'text': body,
                'label': 1 if is_fake else 0,
                'date': date,
                'source_category': np.random.choice(['political', 'health', 'science', 'general'])
            })
        
        return pd.DataFrame(data)
    
    def load_dataset(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load dataset from file or create synthetic one.
        
        Args:
            path: Path to dataset CSV file
            
        Returns:
            Loaded or synthetic DataFrame
        """
        if path and path.exists():
            df = pd.read_csv(path)
            return df
        
        # Create synthetic dataset for demonstration
        print("Creating synthetic dataset for demonstration...")
        return self.create_synthetic_dataset(n_samples=2000)
    
    def deduplicate(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Remove duplicate entries based on text content.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to check for duplicates
            
        Returns:
            Deduplicated DataFrame
        """
        # Create hash of text for efficient deduplication
        df['_text_hash'] = df[text_column].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest() if pd.notna(x) else None
        )
        df_deduped = df.drop_duplicates(subset=['_text_hash']).drop(columns=['_text_hash'])
        
        print(f"Removed {len(df) - len(df_deduped)} duplicate entries")
        return df_deduped
    
    def preprocess_dataset(self, df: pd.DataFrame, 
                           text_columns: List[str] = ['title', 'text'],
                           for_bert: bool = False) -> pd.DataFrame:
        """
        Preprocess all text columns in the dataset.
        
        Args:
            df: Input DataFrame
            text_columns: List of columns to preprocess
            for_bert: Use minimal preprocessing for BERT
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        for col in text_columns:
            if col in df.columns:
                if for_bert:
                    df[f'{col}_processed'] = df[col].apply(
                        self.preprocessor.preprocess_for_bert
                    )
                else:
                    df[f'{col}_processed'] = df[col].apply(
                        lambda x: self.preprocessor.preprocess(x, remove_stops=True, lemmatize=True)
                    )
        
        # Create combined text field
        if 'title' in df.columns and 'text' in df.columns:
            title_col = 'title_processed' if 'title_processed' in df.columns else 'title'
            text_col = 'text_processed' if 'text_processed' in df.columns else 'text'
            df['combined_text'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
        
        return df
    
    def temporal_split(self, df: pd.DataFrame, 
                       date_column: str = 'date',
                       train_cutoff: str = '2023-06-01',
                       val_cutoff: str = '2024-01-01') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset temporally for realistic evaluation.
        
        Args:
            df: Input DataFrame
            date_column: Column containing dates
            train_cutoff: End date for training data
            val_cutoff: End date for validation data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        train_cutoff = pd.to_datetime(train_cutoff)
        val_cutoff = pd.to_datetime(val_cutoff)
        
        train_df = df[df[date_column] < train_cutoff]
        val_df = df[(df[date_column] >= train_cutoff) & (df[date_column] < val_cutoff)]
        test_df = df[df[date_column] >= val_cutoff]
        
        print(f"Temporal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def random_split(self, df: pd.DataFrame, 
                     stratify_column: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Random stratified split of dataset.
        
        Args:
            df: Input DataFrame
            stratify_column: Column to stratify by
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df, temp_df = train_test_split(
            df, 
            test_size=(TEST_SIZE + VAL_SIZE),
            random_state=RANDOM_SEED,
            stratify=df[stratify_column] if stratify_column in df.columns else None
        )
        
        val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_ratio,
            random_state=RANDOM_SEED,
            stratify=temp_df[stratify_column] if stratify_column in temp_df.columns else None
        )
        
        print(f"Random split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, 
                            val_df: pd.DataFrame, 
                            test_df: pd.DataFrame,
                            prefix: str = 'processed') -> Dict[str, Path]:
        """
        Save processed datasets to files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            prefix: Filename prefix
            
        Returns:
            Dictionary with paths to saved files
        """
        paths = {}
        
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            path = PROCESSED_DATA_DIR / f'{prefix}_{name}.csv'
            df.to_csv(path, index=False)
            paths[name] = path
            print(f"Saved {name} data to {path}")
        
        return paths


def prepare_data(use_synthetic: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to prepare data for training.
    
    Args:
        use_synthetic: Whether to use synthetic data for demonstration
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    loader = DatasetLoader()
    
    # Load or create dataset
    if use_synthetic:
        df = loader.create_synthetic_dataset(n_samples=2000)
    else:
        df = loader.load_dataset()
    
    # Deduplicate
    df = loader.deduplicate(df)
    
    # Preprocess
    df = loader.preprocess_dataset(df)
    
    # Split
    train_df, val_df, test_df = loader.random_split(df)
    
    # Save
    loader.save_processed_data(train_df, val_df, test_df)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = prepare_data()
    print("\nData preparation complete!")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
