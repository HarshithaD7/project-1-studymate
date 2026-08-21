"""
One-time local setup script -- run this on your own machine, not in any
sandboxed/CI environment, since it needs real internet access to
huggingface.co.

Why this exists: on Streamlit Community Cloud, get_embeddings() in
src/rag_service.py was failing with "couldn't connect to huggingface.co
... and couldn't find them in the cached files" -- the model was never
cached in that container, and the online download attempt itself was
also failing (not just slow). Bundling the model's files directly into
the repo removes the huggingface.co network dependency at runtime
entirely: get_embeddings() already checks for this local folder first
and uses it if present, with no network call at all.

Usage:
    pip install sentence-transformers --break-system-packages   # if not already installed
    python scripts/download_embedding_model.py

This saves the model into models/all-MiniLM-L6-v2/ (relative to the
project root). After it finishes, commit that folder to git -- see the
printed instructions at the end of this script.
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(PROJECT_DIR, "models", "all-MiniLM-L6-v2")


def main():

    if os.path.isdir(TARGET_DIR) and os.listdir(TARGET_DIR):
        print(f"Already present: {TARGET_DIR}")
        print("Delete that folder first if you want to re-download.")
        return

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "sentence-transformers isn't installed in this environment.\n"
            "Run: pip install sentence-transformers --break-system-packages\n"
            "then re-run this script."
        )
        sys.exit(1)

    print("Downloading sentence-transformers/all-MiniLM-L6-v2 from Hugging Face...")
    print("(This needs real internet access -- run it on your own machine,")
    print(" not inside a restricted sandbox.)")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    os.makedirs(os.path.dirname(TARGET_DIR), exist_ok=True)
    model.save(TARGET_DIR)

    total_size_mb = sum(
        os.path.getsize(os.path.join(TARGET_DIR, f))
        for f in os.listdir(TARGET_DIR)
        if os.path.isfile(os.path.join(TARGET_DIR, f))
    ) / (1024 * 1024)

    print(f"\nSaved to: {TARGET_DIR}")
    print(f"Total size: {total_size_mb:.1f} MB")
    print("\nNext steps:")
    print("  git add models/all-MiniLM-L6-v2")
    print('  git commit -m "Bundle embedding model to remove huggingface.co runtime dependency"')
    print("  git push origin main")


if __name__ == "__main__":
    main()
