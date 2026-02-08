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


def train_models(tune_hyperparams: bool = False):
    """Train the single TF-IDF + Logistic Regression model."""
    print("=" * 60)
    print("MISINFORMATION DETECTION - MODEL TRAINING (SINGLE MODEL)")
    print("=" * 60)
    from src.data_preprocessing import prepare_data
    from src.traditional_ml import train_single_model
    print("\n[1/2] Preparing data...")
    train_df, val_df, test_df = prepare_data()
    print("\n[2/2] Training TF-IDF + Logistic Regression...")
    results = train_single_model(train_df, val_df, tune_hyperparams=tune_hyperparams)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    return results, test_df


def evaluate_models(test_df=None):
    """Evaluate the single trained model (TF-IDF + LR)."""
    print("=" * 60)
    print("MISINFORMATION DETECTION - MODEL EVALUATION")
    print("=" * 60)
    if test_df is None:
        from src.data_preprocessing import prepare_data
        _, _, test_df = prepare_data()
    from src.evaluation import ModelEvaluator
    from src.traditional_ml import TfidfLogisticClassifier
    from src.config import MODELS_DIR
    evaluator = ModelEvaluator()
    tfidf_path = MODELS_DIR / 'tfidf_logistic.pkl'
    if not tfidf_path.exists():
        print("\nNo trained model found. Run: python main.py --train")
        return None
    print("\nEvaluating TF-IDF + Logistic Regression...")
    tfidf_lr = TfidfLogisticClassifier()
    tfidf_lr.load(tfidf_path)
    y_pred = tfidf_lr.predict(test_df['combined_text'])
    y_proba = tfidf_lr.predict_proba(test_df['combined_text'])[:, 1]
    latency = tfidf_lr.measure_inference_latency(test_df['combined_text'])
    evaluator.evaluate_model('TF-IDF + LR', test_df['label'].values, y_pred, y_proba, latency)
    print("\nGenerating evaluation visualizations...")
    evaluator.plot_metrics_comparison()
    evaluator.plot_latency_comparison()
    evaluator.plot_confusion_matrix('TF-IDF + LR')
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
  python main.py --train                    Train TF-IDF + LR model
  python main.py --train --tune             Train with hyperparameter tuning
  python main.py --evaluate                 Evaluate trained model
  python main.py --demo                     Run inference demo
  python main.py --api                      Start API server
  python main.py --all                      Train, evaluate, and run demo
        """
    )
    
    parser.add_argument('--train', action='store_true',
                        help='Train the single TF-IDF + LR model')
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
        _, test_df = train_models(tune_hyperparams=args.tune)
    
    if args.all or args.evaluate:
        evaluate_models(test_df)
    
    if args.all or args.demo:
        run_inference_demo()
    
    if args.api:
        run_api_server(port=args.port)


if __name__ == "__main__":
    main()
