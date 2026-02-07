"""
Traditional Machine Learning baselines for misinformation detection.
Implements Naive Bayes and TF-IDF + Logistic Regression models.
"""

import pickle
import time
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from src.config import (
    MODELS_DIR, TFIDF_MAX_FEATURES, NGRAM_RANGE,
    TFIDF_MIN_DF, TFIDF_MAX_DF, LR_MAX_ITER,
    RANDOM_SEED, LABEL_NAMES
)


class TraditionalMLClassifier:
    """Base class for traditional ML classifiers."""
    
    def __init__(self, model_name: str = "baseline"):
        self.model_name = model_name
        self.pipeline = None
        self.is_trained = False
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'TraditionalMLClassifier':
        """Train the model."""
        raise NotImplementedError
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Predict labels for input texts."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Predict probability scores for input texts."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='macro'),
            'recall': recall_score(y, y_pred, average='macro'),
            'f1_score': f1_score(y, y_pred, average='macro'),
            'roc_auc': roc_auc_score(y, y_proba),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, target_names=list(LABEL_NAMES.values()))
        }
        
        return metrics
    
    def measure_inference_latency(self, X: pd.Series, n_samples: int = 100) -> Dict[str, float]:
        """
        Measure inference latency.
        
        Args:
            X: Input texts
            n_samples: Number of samples to test
            
        Returns:
            Dictionary with latency statistics
        """
        sample = X.head(n_samples)
        
        times = []
        for text in sample:
            start = time.time()
            self.pipeline.predict([text])
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        return {
            'mean_latency_ms': np.mean(times),
            'std_latency_ms': np.std(times),
            'max_latency_ms': np.max(times),
            'min_latency_ms': np.min(times)
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / f'{self.model_name}.pkl'
        
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        print(f"Model saved to {path}")
        return path
    
    def load(self, path: Optional[Path] = None) -> 'TraditionalMLClassifier':
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / f'{self.model_name}.pkl'
        
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        self.is_trained = True
        print(f"Model loaded from {path}")
        return self


class NaiveBayesClassifier(TraditionalMLClassifier):
    """Multinomial Naive Bayes classifier with Count Vectorizer."""
    
    def __init__(self, max_features: int = TFIDF_MAX_FEATURES, 
                 ngram_range: Tuple[int, int] = NGRAM_RANGE):
        super().__init__(model_name="naive_bayes")
        self.max_features = max_features
        self.ngram_range = ngram_range
    
    def fit(self, X: pd.Series, y: pd.Series, 
            tune_hyperparams: bool = False) -> 'NaiveBayesClassifier':
        """
        Train Naive Bayes model.
        
        Args:
            X: Input texts
            y: Labels
            tune_hyperparams: Whether to perform grid search
            
        Returns:
            Trained classifier
        """
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                min_df=TFIDF_MIN_DF,
                max_df=TFIDF_MAX_DF
            )),
            ('classifier', MultinomialNB(alpha=0.5))
        ])
        
        if tune_hyperparams:
            param_grid = {
                'vectorizer__max_features': [15000, 25000, 35000],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
            }
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=0
            )
            grid_search.fit(X, y)
            self.pipeline = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.pipeline.fit(X, y)
        
        self.is_trained = True
        return self


class TfidfLogisticClassifier(TraditionalMLClassifier):
    """TF-IDF + Logistic Regression classifier."""
    
    def __init__(self, max_features: int = TFIDF_MAX_FEATURES,
                 ngram_range: Tuple[int, int] = NGRAM_RANGE):
        super().__init__(model_name="tfidf_logistic")
        self.max_features = max_features
        self.ngram_range = ngram_range
    
    def fit(self, X: pd.Series, y: pd.Series,
            tune_hyperparams: bool = False) -> 'TfidfLogisticClassifier':
        """
        Train TF-IDF + Logistic Regression model.
        
        Args:
            X: Input texts
            y: Labels
            tune_hyperparams: Whether to perform grid search
            
        Returns:
            Trained classifier
        """
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                sublinear_tf=True,
                min_df=TFIDF_MIN_DF,
                max_df=TFIDF_MAX_DF
            )),
            ('classifier', LogisticRegression(
                random_state=RANDOM_SEED,
                max_iter=LR_MAX_ITER,
                class_weight='balanced',
                C=1.0,
                solver='lbfgs'
            ))
        ])
        
        if tune_hyperparams:
            param_grid = {
                'vectorizer__max_features': [15000, 25000, 35000],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'classifier__C': [0.5, 1.0, 2.0, 5.0],
                'classifier__penalty': ['l2']
            }
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=0
            )
            grid_search.fit(X, y)
            self.pipeline = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.pipeline.fit(X, y)
        
        self.is_trained = True
        return self
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, np.ndarray]:
        """
        Get most important features for each class.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        vectorizer = self.pipeline.named_steps['vectorizer']
        classifier = self.pipeline.named_steps['classifier']
        
        feature_names = np.array(vectorizer.get_feature_names_out())
        coef = classifier.coef_[0]
        
        # Get top features for misinformation (positive coefficients)
        top_misinfo_idx = np.argsort(coef)[-top_n:][::-1]
        top_misinfo_features = feature_names[top_misinfo_idx]
        top_misinfo_scores = coef[top_misinfo_idx]
        
        # Get top features for credible (negative coefficients)
        top_credible_idx = np.argsort(coef)[:top_n]
        top_credible_features = feature_names[top_credible_idx]
        top_credible_scores = coef[top_credible_idx]
        
        return {
            'misinformation_features': top_misinfo_features,
            'misinformation_scores': top_misinfo_scores,
            'credible_features': top_credible_features,
            'credible_scores': top_credible_scores
        }


class MajorityClassifier:
    """Naive baseline that always predicts the majority class."""
    
    def __init__(self):
        self.majority_class = None
        self.class_prob = None
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'MajorityClassifier':
        """Train by finding majority class."""
        class_counts = y.value_counts()
        self.majority_class = class_counts.idxmax()
        self.class_prob = class_counts[self.majority_class] / len(y)
        return self
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Predict majority class for all inputs."""
        return np.full(len(X), self.majority_class)
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Return probability based on class distribution."""
        probs = np.zeros((len(X), 2))
        probs[:, self.majority_class] = self.class_prob
        probs[:, 1 - self.majority_class] = 1 - self.class_prob
        return probs
    
    def evaluate(self, X: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }


def train_traditional_baselines(train_df: pd.DataFrame, 
                                val_df: pd.DataFrame,
                                text_column: str = 'combined_text',
                                label_column: str = 'label',
                                tune_hyperparams: bool = False) -> Dict[str, Any]:
    """
    Train all traditional ML baselines.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        text_column: Column with input text
        label_column: Column with labels
        tune_hyperparams: Whether to tune hyperparameters
        
    Returns:
        Dictionary with trained models and their metrics
    """
    X_train = train_df[text_column]
    y_train = train_df[label_column]
    X_val = val_df[text_column]
    y_val = val_df[label_column]
    
    results = {}
    
    # Majority baseline
    print("\n=== Training Majority Baseline ===")
    majority = MajorityClassifier()
    majority.fit(X_train, y_train)
    results['majority'] = {
        'model': majority,
        'metrics': majority.evaluate(X_val, y_val)
    }
    print(f"Majority Baseline F1: {results['majority']['metrics']['f1_score']:.4f}")
    
    # Naive Bayes
    print("\n=== Training Naive Bayes ===")
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train, tune_hyperparams=tune_hyperparams)
    results['naive_bayes'] = {
        'model': nb,
        'metrics': nb.evaluate(X_val, y_val),
        'latency': nb.measure_inference_latency(X_val)
    }
    print(f"Naive Bayes F1: {results['naive_bayes']['metrics']['f1_score']:.4f}")
    print(f"Mean Latency: {results['naive_bayes']['latency']['mean_latency_ms']:.2f}ms")
    
    # TF-IDF + Logistic Regression
    print("\n=== Training TF-IDF + Logistic Regression ===")
    tfidf_lr = TfidfLogisticClassifier()
    tfidf_lr.fit(X_train, y_train, tune_hyperparams=tune_hyperparams)
    results['tfidf_logistic'] = {
        'model': tfidf_lr,
        'metrics': tfidf_lr.evaluate(X_val, y_val),
        'latency': tfidf_lr.measure_inference_latency(X_val),
        'feature_importance': tfidf_lr.get_feature_importance()
    }
    print(f"TF-IDF + LR F1: {results['tfidf_logistic']['metrics']['f1_score']:.4f}")
    print(f"Mean Latency: {results['tfidf_logistic']['latency']['mean_latency_ms']:.2f}ms")
    
    # Print feature importance
    fi = results['tfidf_logistic']['feature_importance']
    print("\nTop features for Misinformation:")
    for feat, score in zip(fi['misinformation_features'][:10], fi['misinformation_scores'][:10]):
        print(f"  {feat}: {score:.4f}")
    
    print("\nTop features for Credible:")
    for feat, score in zip(fi['credible_features'][:10], fi['credible_scores'][:10]):
        print(f"  {feat}: {score:.4f}")
    
    # Save models
    nb.save()
    tfidf_lr.save()
    
    return results


if __name__ == "__main__":
    from src.data_preprocessing import prepare_data
    
    print("Preparing data...")
    train_df, val_df, test_df = prepare_data()
    
    print("\nTraining traditional ML baselines...")
    results = train_traditional_baselines(train_df, val_df)
    
    print("\n=== Final Comparison ===")
    for model_name, result in results.items():
        print(f"{model_name}: F1={result['metrics']['f1_score']:.4f}")
