"""
Main runner script for the Misinformation Detection System.
Provides a unified interface for training, evaluation, and inference.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def train_models(train_bert: bool = False, tune_hyperparams: bool = False):
    """Train all models."""
    print("=" * 60)
    print("MISINFORMATION DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    from src.data_preprocessing import prepare_data
    from src.traditional_ml import train_traditional_baselines
    
    print("\n[1/3] Preparing data...")
    train_df, val_df, test_df = prepare_data()
    
    print("\n[2/3] Training traditional ML baselines...")
    traditional_results = train_traditional_baselines(
        train_df, val_df, 
        tune_hyperparams=tune_hyperparams
    )
    
    bert_results = None
    if train_bert:
        print("\n[3/3] Training BERT model...")
        from src.deep_learning import train_bert_model
        bert_results = train_bert_model(train_df, val_df, num_epochs=3, batch_size=8)
    else:
        print("\n[3/3] Skipping BERT training (use --train-bert to enable)")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return traditional_results, bert_results, test_df


def evaluate_models(test_df=None):
    """Evaluate all trained models."""
    print("=" * 60)
    print("MISINFORMATION DETECTION - MODEL EVALUATION")
    print("=" * 60)
    
    if test_df is None:
        from src.data_preprocessing import prepare_data
        _, _, test_df = prepare_data()
    
    from src.evaluation import ModelEvaluator
    from src.traditional_ml import NaiveBayesClassifier, TfidfLogisticClassifier
    from src.config import MODELS_DIR
    
    evaluator = ModelEvaluator()
    models = {}
    
    # Load and evaluate Naive Bayes
    nb_path = MODELS_DIR / 'naive_bayes.pkl'
    if nb_path.exists():
        print("\nEvaluating Naive Bayes...")
        nb = NaiveBayesClassifier()
        nb.load(nb_path)
        
        y_pred = nb.predict(test_df['combined_text'])
        y_proba = nb.predict_proba(test_df['combined_text'])[:, 1]
        latency = nb.measure_inference_latency(test_df['combined_text'])
        
        evaluator.evaluate_model('Naive Bayes', test_df['label'].values, y_pred, y_proba, latency)
        models['Naive Bayes'] = {'model': nb, 'latency': latency}
    
    # Load and evaluate TF-IDF + Logistic Regression
    tfidf_path = MODELS_DIR / 'tfidf_logistic.pkl'
    if tfidf_path.exists():
        print("\nEvaluating TF-IDF + Logistic Regression...")
        tfidf_lr = TfidfLogisticClassifier()
        tfidf_lr.load(tfidf_path)
        
        y_pred = tfidf_lr.predict(test_df['combined_text'])
        y_proba = tfidf_lr.predict_proba(test_df['combined_text'])[:, 1]
        latency = tfidf_lr.measure_inference_latency(test_df['combined_text'])
        
        evaluator.evaluate_model('TF-IDF + LR', test_df['label'].values, y_pred, y_proba, latency)
        models['TF-IDF + LR'] = {'model': tfidf_lr, 'latency': latency}
    
    # Load and evaluate BERT
    bert_path = MODELS_DIR / 'bert_model'
    if bert_path.exists():
        print("\nEvaluating BERT model...")
        from src.deep_learning import BertClassifier
        bert = BertClassifier()
        bert.load(bert_path)
        
        y_pred, y_proba_full = bert.predict_batch(test_df['combined_text'].tolist())
        y_proba = y_proba_full[:, 1]
        latency = bert.measure_inference_latency(test_df['combined_text'].tolist())
        
        evaluator.evaluate_model('DistilBERT', test_df['label'].values, y_pred, y_proba, latency)
        models['DistilBERT'] = {'model': bert, 'latency': latency}
    
    if not models:
        print("\nNo trained models found. Run training first.")
        return None
    
    # Generate visualizations and report
    print("\nGenerating evaluation visualizations...")
    evaluator.plot_metrics_comparison()
    evaluator.plot_latency_comparison()
    
    for model_name in models.keys():
        evaluator.plot_confusion_matrix(model_name)
    
    report = evaluator.generate_report()
    evaluator.save_results()
    
    print("\n" + report)
    
    return evaluator


def run_inference_demo():
    """Run interactive inference demo."""
    print("=" * 60)
    print("MISINFORMATION DETECTION - INFERENCE DEMO")
    print("=" * 60)
    
    from src.inference import load_models, InferenceEngine
    
    models = load_models()
    if not models:
        print("\nNo models found. Please train models first.")
        return
    
    engine = InferenceEngine(models)
    
    # Demo texts
    demo_texts = [
        "BREAKING: You won't believe what they're hiding about vaccines!!!",
        "According to official sources, the new policy has been confirmed by multiple experts.",
        "SHOCKING REVELATION: The truth about climate change they don't want you to know!",
        "A new study published in Nature reveals findings about climate change research."
    ]
    
    print("\nRunning inference on demo texts...\n")
    
    for i, text in enumerate(demo_texts, 1):
        print(f"[{i}] Text: {text[:80]}...")
        result = engine.predict(text)
        print(f"    Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
        print(f"    Latency: {result['latency_ms']:.2f}ms")
        print()
    
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


def run_api_server(port: int = 5000):
    """Start the inference API server."""
    print("=" * 60)
    print("MISINFORMATION DETECTION - API SERVER")
    print("=" * 60)
    
    from src.inference import run_api
    run_api(port=port)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Misinformation Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                    Train traditional ML models
  python main.py --train --train-bert       Train all models including BERT
  python main.py --evaluate                 Evaluate trained models
  python main.py --demo                     Run inference demo
  python main.py --api                      Start API server
  python main.py --all                      Train, evaluate, and run demo
        """
    )
    
    parser.add_argument('--train', action='store_true',
                        help='Train models')
    parser.add_argument('--train-bert', action='store_true',
                        help='Include BERT in training (requires GPU for speed)')
    parser.add_argument('--tune', action='store_true',
                        help='Tune hyperparameters during training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained models')
    parser.add_argument('--demo', action='store_true',
                        help='Run inference demo')
    parser.add_argument('--api', action='store_true',
                        help='Start API server')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for API server (default: 5000)')
    parser.add_argument('--all', action='store_true',
                        help='Run full pipeline: train, evaluate, demo')
    
    args = parser.parse_args()
    
    # Default to showing help if no arguments
    if not any([args.train, args.evaluate, args.demo, args.api, args.all]):
        parser.print_help()
        return
    
    test_df = None
    
    if args.all or args.train:
        _, _, test_df = train_models(
            train_bert=args.train_bert,
            tune_hyperparams=args.tune
        )
    
    if args.all or args.evaluate:
        evaluate_models(test_df)
    
    if args.all or args.demo:
        run_inference_demo()
    
    if args.api:
        run_api_server(port=args.port)


if __name__ == "__main__":
    main()
