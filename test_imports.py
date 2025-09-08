import sys

print("--- Starting import test ---", file=sys.stderr)

try:
    print("Importing sys... SUCCESS", file=sys.stderr)
    import gradio
    print("Importing gradio... SUCCESS", file=sys.stderr)
    import torch
    print("Importing torch... SUCCESS", file=sys.stderr)
    import transformers
    print("Importing transformers... SUCCESS", file=sys.stderr)
    print("--- All imports successful ---", file=sys.stderr)
except Exception as e:
    print(f"--- An error occurred during import ---", file=sys.stderr)
    print(e, file=sys.stderr)
