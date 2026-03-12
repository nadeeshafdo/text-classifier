"""
CLI entry point for the text classifier.

Usage:
    python main.py train
    python main.py predict "your project idea text"
    python main.py predict --interactive
"""

import sys


def main():
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        _handle_train()
    elif command == "predict":
        _handle_predict()
    else:
        print(f"Unknown command: {command}")
        _print_usage()
        sys.exit(1)


def _handle_train():
    from src.train import train
    train()


def _handle_predict():
    from src.predict import predict

    # Check for --interactive flag
    if len(sys.argv) > 2 and sys.argv[2] == "--interactive":
        _interactive_mode(predict)
        return

    # Single prediction from command line argument
    if len(sys.argv) < 3:
        print("Error: provide text to classify or use --interactive")
        print('  python main.py predict "your idea here"')
        print("  python main.py predict --interactive")
        sys.exit(1)

    text = " ".join(sys.argv[2:])
    _print_prediction(predict(text))


def _interactive_mode(predict_fn):
    print("=== Interactive Mode (type 'quit' to exit) ===\n")
    while True:
        try:
            text = input("Enter project idea: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        if not text:
            continue

        _print_prediction(predict_fn(text))
        print()


def _print_prediction(results: list[dict], top_n: int = 10):
    """Print prediction results in a readable format."""
    print(f"\n{'Domain':<35} {'Confidence':>10}")
    print("-" * 47)
    for item in results[:top_n]:
        conf = item["confidence"]
        bar = "█" * int(conf * 20)
        print(f"  {item['domain']:<33} {conf:>8.2%}  {bar}")

    if len(results) > top_n:
        print(f"  ... and {len(results) - top_n} more domains with lower confidence")


def _print_usage():
    print(__doc__)


if __name__ == "__main__":
    main()
