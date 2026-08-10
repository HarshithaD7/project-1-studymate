import os
from dotenv import load_dotenv

from vectorize_book import (
    vectorize_ncert,
    vectorize_pyqs,
    vectorize_model_answers
)

load_dotenv()

print("\n==============================")
print("BIOASSIST VECTORIZATION")
print("==============================\n")

try:
    vectorize_ncert("class_11")
    vectorize_ncert("class_12")
    vectorize_pyqs()
    vectorize_model_answers()

    print("\n✅ All vectorization completed!")

except Exception as e:
    print(f"❌ Vectorization error: {e}")
