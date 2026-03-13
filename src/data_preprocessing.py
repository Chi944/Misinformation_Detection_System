"""
Data acquisition, cleaning, and preprocessing module.
Uses Hugging Face: load_dataset("kasperdinh/fake-news-detection") plus data in data/raw/.
Handles HTML parsing, deduplication, and train/val/test splitting.
"""

import hashlib
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TEST_SIZE,
    VAL_SIZE,
)

warnings.filterwarnings("ignore")

# Files to look for in project root and copy to data/raw
RAW_DATA_FILES = ["FakeNewsNet.csv", "train.tsv", "valid.tsv", "test.tsv"]


def organise_data_folders() -> None:
    """
    Copy raw data from project root into data/raw so the loader finds them.
    Idempotent: only copies if source exists and target is missing or older.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in RAW_DATA_FILES:
        src = PROJECT_ROOT / name
        dst = RAW_DATA_DIR / name
        if src.exists() and src.is_file():
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                import shutil

                shutil.copy2(src, dst)
                print(f"Organised: {name} -> data/raw/")


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
        # Be resilient when running offline / behind restricted networks:
        # NLTK downloads can fail, and missing corpora would otherwise crash init.
        self.stop_words = set()
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                self.stop_words = set(stopwords.words("english"))
            except Exception:
                self.stop_words = set()

    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        packages = ["punkt", "stopwords", "wordnet", "punkt_tab"]
        for package in packages:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass

    def clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if pd.isna(text):
            return ""
        clean = re.sub(r"<[^>]+>", "", str(text))
        return clean

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = (
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        return re.sub(url_pattern, "", text)

    def remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and spaces."""
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        return " ".join(text.split())

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
        try:
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        except LookupError:
            # WordNet resource may be missing; try to fetch it and retry once.
            try:
                nltk.download("wordnet", quiet=True)
                return [self.lemmatizer.lemmatize(t) for t in tokens]
            except Exception:
                # Safe fallback: skip lemmatization if resources unavailable.
                return tokens

    def preprocess(self, text: str, remove_stops: bool = False, lemmatize: bool = False) -> str:
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
            text = " ".join(tokens)

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
            "Experts interviewed by major outlets agree that {topic} is significant.",
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
            "ALERT: {topic} is being censored! Share now!",
        ]

        topics = [
            "the new policy",
            "vaccine effectiveness",
            "election results",
            "climate change data",
            "economic forecasts",
            "health guidelines",
            "scientific research",
            "government spending",
            "international relations",
            "technology developments",
            "environmental regulations",
            "public safety measures",
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
                    institution=np.random.choice(institutions),
                )
                body = f"{title} Additional details support this conclusion. " * np.random.randint(
                    2, 5
                )

            # Generate fake dates for temporal splitting
            year = np.random.choice([2022, 2023, 2024], p=[0.4, 0.4, 0.2])
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            date = f"{year}-{month:02d}-{day:02d}"

            data.append(
                {
                    "id": i,
                    "title": title,
                    "text": body,
                    "label": 1 if is_fake else 0,
                    "date": date,
                    "source_category": np.random.choice(
                        ["political", "health", "science", "general"]
                    ),
                }
            )

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

    def load_from_huggingface(
        self, dataset_name: str = "kasperdinh/fake-news-detection"
    ) -> pd.DataFrame:
        """
        Load dataset from Hugging Face datasets.

        Args:
            dataset_name: Hugging Face dataset identifier

        Returns:
            DataFrame with title, text, label (0=Credible, 1=Misinformation)
        """
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        print(f"Loading Hugging Face dataset: {dataset_name}")
        ds = hf_load_dataset(dataset_name)

        # Handle split structure (e.g. train, test, validation)
        dfs = []
        for split in ds.keys():
            dfs.append(ds[split].to_pandas())
        df = pd.concat(dfs, ignore_index=True)

        return self._normalize_hf_df(df)

    def load_from_fakenewsnet(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load FakeNewsNet.csv format (title, news_url, source_domain, tweet_num, real).
        real=1 means credible, real=0 means misinformation.

        Args:
            path: Path to FakeNewsNet.csv (default: project root or data/raw)

        Returns:
            DataFrame with title, text, label (0=Credible, 1=Misinformation)
        """
        if path is None:
            path = PROJECT_ROOT / "FakeNewsNet.csv"
            if not path.exists():
                path = RAW_DATA_DIR / "FakeNewsNet.csv"

        if not path.exists():
            raise FileNotFoundError(f"FakeNewsNet.csv not found at {path}")

        print(f"Loading FakeNewsNet from {path}")
        df = pd.read_csv(path)

        # Normalize: real=1 -> label=0 (Credible), real=0 -> label=1 (Misinformation)
        if "real" in df.columns:
            df["label"] = (1 - df["real"]).astype(int)
        else:
            raise ValueError("FakeNewsNet must have 'real' column")

        if "title" in df.columns:
            df["text"] = df["title"].fillna("")
        else:
            raise ValueError("FakeNewsNet must have 'title' column")

        return df[["title", "text", "label"]].copy()

    def load_from_tsv(self, path: Path) -> pd.DataFrame:
        """
        Load a single TSV file (e.g. Liar-style: id, label_str, statement, ...).
        Maps label: true, mostly-true -> 0 (Credible); false, pants-fire, barely-true, half-true -> 1 (Misinformation).
        """
        df = pd.read_csv(path, sep="\t", header=None, on_bad_lines="skip")
        if df.shape[1] < 3:
            raise ValueError(
                f"TSV must have at least 3 columns (id, label, text). Got {df.shape[1]}"
            )
        # Column 0: id, 1: label (string), 2: statement/text
        df = df.rename(columns={0: "id", 1: "label_str", 2: "text"})
        df["text"] = df["text"].fillna("").astype(str)
        credible = ["true", "mostly-true"]
        df["label"] = (
            df["label_str"].astype(str).str.lower().map(lambda x: 0 if x in credible else 1)
        )
        df["title"] = df["text"].str[:200]
        return df[["title", "text", "label"]].copy()

    def load_from_data_raw(self) -> List[pd.DataFrame]:
        """
        Load all supported data from data/raw: FakeNewsNet.csv, train.tsv, valid.tsv, test.tsv.
        Returns list of DataFrames (may be empty).
        """
        dfs = []
        # FakeNewsNet
        for name in ["FakeNewsNet.csv", "fake_news_net.csv"]:
            p = RAW_DATA_DIR / name
            if p.exists():
                dfs.append(self.load_from_fakenewsnet(p))
                break
        # TSV files (Liar-style)
        for name in ["train.tsv", "valid.tsv", "test.tsv"]:
            p = RAW_DATA_DIR / name
            if p.exists():
                try:
                    dfs.append(self.load_from_tsv(p))
                except Exception as e:
                    print(f"Warning: Could not load {name}: {e}")
        return dfs

    def _normalize_hf_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize Hugging Face dataset to title, text, label format.
        Handles common column names: text, title, label, fake, real, content, etc.
        """
        out = pd.DataFrame()
        cols = {c.lower(): c for c in df.columns}

        # Resolve label column (0=Credible, 1=Misinformation)
        if "label" in cols:
            out["label"] = df[cols["label"]].astype(int)
        elif "fake" in cols:
            out["label"] = df[cols["fake"]].astype(int)
        elif "real" in cols:
            out["label"] = 1 - df[cols["real"]].astype(int)
        else:
            raise ValueError(
                f"Dataset must have 'label', 'fake', or 'real' column. Found: {list(df.columns)}"
            )

        # Resolve text column
        if "text" in cols:
            out["text"] = df[cols["text"]].fillna("").astype(str)
        elif "content" in cols:
            out["text"] = df[cols["content"]].fillna("").astype(str)
        elif "article" in cols:
            out["text"] = df[cols["article"]].fillna("").astype(str)
        else:
            out["text"] = ""

        # Resolve title column
        if "title" in cols:
            out["title"] = df[cols["title"]].fillna("").astype(str)
        else:
            out["title"] = out["text"].str[:200]

        return out

    def clean_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize combined dataset.
        - Drop rows with empty text
        - Normalize label to 0/1
        - Remove duplicates
        """
        df = df.copy()

        if "text" not in df.columns and "title" in df.columns:
            df["text"] = df["title"].fillna("")
        if "text" not in df.columns:
            raise ValueError("DataFrame must have 'text' or 'title' column")

        # Drop rows with empty or too-short text
        df["text"] = df["text"].fillna("").astype(str)
        df = df[df["text"].str.strip().str.len() >= 10].copy()

        # Ensure label is 0 or 1
        if "label" in df.columns:
            df["label"] = df["label"].astype(int).clip(0, 1)

        if "title" not in df.columns:
            df["title"] = df["text"].str[:200]

        print(f"Cleaned: {len(df)} rows")
        return df

    def deduplicate(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Remove duplicate entries based on text content.

        Args:
            df: Input DataFrame
            text_column: Column containing text to check for duplicates

        Returns:
            Deduplicated DataFrame
        """
        # Create hash of text for efficient deduplication
        df["_text_hash"] = df[text_column].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest() if pd.notna(x) else None
        )
        df_deduped = df.drop_duplicates(subset=["_text_hash"]).drop(columns=["_text_hash"])

        print(f"Removed {len(df) - len(df_deduped)} duplicate entries")
        return df_deduped

    def preprocess_dataset(
        self, df: pd.DataFrame, text_columns: List[str] = ["title", "text"], for_bert: bool = False
    ) -> pd.DataFrame:
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

        # Ensure text column exists (for title-only datasets like FakeNewsNet)
        if "text" not in df.columns and "title" in df.columns:
            df["text"] = df["title"].fillna("")

        for col in text_columns:
            if col in df.columns:
                if for_bert:
                    df[f"{col}_processed"] = df[col].apply(self.preprocessor.preprocess_for_bert)
                else:
                    df[f"{col}_processed"] = df[col].apply(
                        lambda x: self.preprocessor.preprocess(x, remove_stops=True, lemmatize=True)
                    )

        # Create combined text field (title + text)
        title_col = (
            "title_processed"
            if "title_processed" in df.columns
            else ("title" if "title" in df.columns else None)
        )
        text_col = (
            "text_processed"
            if "text_processed" in df.columns
            else ("text" if "text" in df.columns else None)
        )
        if title_col:
            df["combined_text"] = df[title_col].fillna("").astype(str)
        else:
            df["combined_text"] = ""
        if text_col and text_col != title_col:
            df["combined_text"] = df["combined_text"] + " " + df[text_col].fillna("").astype(str)
        df["combined_text"] = df["combined_text"].str.strip()

        return df

    def temporal_split(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        train_cutoff: str = "2023-06-01",
        val_cutoff: str = "2024-01-01",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        train_cutoff = pd.to_datetime(train_cutoff)
        val_cutoff = pd.to_datetime(val_cutoff)

        train_df = df[df[date_column] < train_cutoff]
        val_df = df[(df[date_column] >= train_cutoff) & (df[date_column] < val_cutoff)]
        test_df = df[df[date_column] >= val_cutoff]

        print(f"Temporal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return train_df, val_df, test_df

    def random_split(
        self, df: pd.DataFrame, stratify_column: str = "label"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            stratify=df[stratify_column] if stratify_column in df.columns else None,
        )

        val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_ratio,
            random_state=RANDOM_SEED,
            stratify=temp_df[stratify_column] if stratify_column in temp_df.columns else None,
        )

        print(f"Random split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return train_df, val_df, test_df

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prefix: str = "processed",
    ) -> Dict[str, Path]:
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

        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            path = PROCESSED_DATA_DIR / f"{prefix}_{name}.csv"
            df.to_csv(path, index=False)
            paths[name] = path
            print(f"Saved {name} data to {path}")

        return paths


def prepare_data(
    use_synthetic: bool = False,
    use_hf: bool = True,
    use_fakenewsnet: bool = True,
    hf_dataset: str = "kasperdinh/fake-news-detection",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to prepare data for training.

    Args:
        use_synthetic: Use synthetic data (skip HF and FakeNewsNet)
        use_hf: Load from Hugging Face datasets (kasperdinh/fake-news-detection)
        use_fakenewsnet: Load from FakeNewsNet.csv (project root or data/raw)
        hf_dataset: Hugging Face dataset identifier

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    organise_data_folders()
    loader = DatasetLoader()
    dfs = []

    if use_synthetic:
        print("Using synthetic dataset...")
        df = loader.create_synthetic_dataset(n_samples=2000)
        dfs.append(df)
    else:
        if use_hf:
            try:
                df_hf = loader.load_from_huggingface(hf_dataset)
                dfs.append(df_hf)
            except Exception as e:
                print(f"Warning: Could not load Hugging Face dataset: {e}")

        # Data in data/raw: FakeNewsNet.csv, train.tsv, valid.tsv, test.tsv
        try:
            raw_dfs = loader.load_from_data_raw()
            dfs.extend(raw_dfs)
        except Exception as e:
            print(f"Warning: load_from_data_raw: {e}")
        # Backward compat: FakeNewsNet in project root only (data/raw already loaded)
        if use_fakenewsnet:
            try:
                root_csv = PROJECT_ROOT / "FakeNewsNet.csv"
                if root_csv.exists():
                    dfs.append(loader.load_from_fakenewsnet(root_csv))
            except Exception as e:
                print(f"Warning: Could not load FakeNewsNet from root: {e}")

        if not dfs:
            print("No data loaded. Falling back to synthetic...")
            dfs.append(loader.create_synthetic_dataset(n_samples=2000))

    df = pd.concat(dfs, ignore_index=True)

    # Clean and normalize
    df = loader.clean_and_normalize(df)

    # Deduplicate
    df = loader.deduplicate(df, text_column="text")

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
