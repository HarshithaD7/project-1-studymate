"""
BioAssist AI -- Capstone evaluation harness.

Runs a fixed, diverse sample of REAL CBSE PYQs through the actual
rag_service.answer_question() and answer_evaluator.evaluate_answer()
functions (the same code path the Streamlit app uses), and produces:

  1. Learn/RAG groundedness results  -> evaluation_results_learn.json
  2. Mark-aware evaluation depth test -> evaluation_results_marks.json
  3. A printed summary table for the Analysis & Results slide.

Run from the project's `src/` folder so the existing local imports
(`from rag_service import ...`) resolve the same way main.py does:

    cd src
    python evaluate_system.py

This makes real Groq API calls using your existing GROQ_API_KEY in
src/.env -- it will take a minute or two and will print progress as
it goes. Nothing here modifies the app; it only reads pyq_questions.json
and calls the existing functions.
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_service import answer_question
from answer_evaluator import evaluate_answer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYQ_PATH = os.path.join(PROJECT_DIR, "pyq_questions.json")

VALID_CHAPTERS = {
    "Sexual Reproduction in Flowering Plants",
    "Human Reproduction",
    "Reproductive Health",
    "Principles of Inheritance and Variation",
    "Molecular Basis of Inheritance",
    "Evolution",
    "Human Health and Diseases",
    "Microbes in Human Welfare",
    "Biotechnology: Principles and Processes",
    "Biotechnology and its Application",
    "Organisms and Populations",
    "Ecosystem",
    "Biodiversity and its Conservation",
}

# rag_service.load_chapter_db() requires the EXACT on-disk vector-DB
# folder name, which is derived straight from the NCERT PDF filename
# (see vectorize_book.py) and always carries a numeric prefix, e.g.
# "1. Sexual Reproduction in Flowering Plants". The chapter names
# above (and the ones stored in pyq_questions.json) are unprefixed,
# so calling answer_question() with r["chapter"] directly always
# missed the DB path -- this is why every question previously came
# back with retrieved=0. Map unprefixed -> real folder name here.
CHAPTER_TO_FOLDER = {
    "Sexual Reproduction in Flowering Plants": "1. Sexual Reproduction in Flowering Plants",
    "Human Reproduction": "2. Human Reproduction",
    "Reproductive Health": "3. Reproductive Health",
    "Principles of Inheritance and Variation": "4. Principles of Inheritance and Variation",
    "Molecular Basis of Inheritance": "5. Molecular Basis of Inheritance",
    "Evolution": "6. Evolution",
    "Human Health and Diseases": "7. Human Health and Diseases",
    "Microbes in Human Welfare": "8. Microbes in Human Welfare",
    "Biotechnology: Principles and Processes": "9. Biotechnology - Principles and Processes",
    "Biotechnology and its Application": "10. Biotechnology and its Application",
    "Organisms and Populations": "11. Organisms and Populations",
    "Ecosystem": "12. Ecosystem",
    "Biodiversity and its Conservation": "13. Biodiversity and its Conservation",
}


def word_overlap_ratio(answer_text, docs):
    """Rough automated groundedness proxy: fraction of the answer's
    meaningful words that also appear somewhere in the retrieved
    NCERT chunks. Not a substitute for reading the output yourself,
    but useful as a quick numeric signal across many questions."""
    import re
    context = " ".join(getattr(d, "page_content", "") for d in docs).lower()
    context_words = set(re.findall(r"[a-z]{4,}", context))
    answer_words = re.findall(r"[a-z]{4,}", answer_text.lower())
    if not answer_words:
        return 0.0
    supported = sum(1 for w in answer_words if w in context_words)
    return round(supported / len(answer_words), 3)


def run_learn_eval(sample_size=12):
    records = json.load(open(PYQ_PATH))
    candidates = [
        r for r in records
        if r.get("chapter") in VALID_CHAPTERS
        and r.get("question_type") in ("MCQ", "Short Answer", "Long Answer")
        and r.get("question") and len(r["question"]) > 20
    ]
    random.seed(7)
    random.shuffle(candidates)

    chosen, per_type = [], {}
    for r in candidates:
        t = r["question_type"]
        if per_type.get(t, 0) >= 5:
            continue
        chosen.append(r)
        per_type[t] = per_type.get(t, 0) + 1
        if len(chosen) >= sample_size:
            break

    print(f"\n=== LEARN / RAG GROUNDEDNESS: {len(chosen)} real PYQs ===\n")
    results = []
    for r in chosen:
        try:
            answer, docs = answer_question(
                r["question"], "Class 12",
                CHAPTER_TO_FOLDER.get(r["chapter"], r["chapter"]),
                explanation_level="Class 12"
            )
            overlap = word_overlap_ratio(answer, docs)
            n_docs = len(docs)
            print(f"[{r['question_type']} | {r['marks']} marks | {r['chapter']}] "
                  f"retrieved={n_docs} overlap={overlap}")
            print(f"  Q: {r['question'][:90]}")
            print(f"  A: {answer[:160].replace(chr(10), ' ')}...\n")
            results.append({
                "chapter": r["chapter"], "type": r["question_type"], "marks": r["marks"],
                "question": r["question"], "answer": answer,
                "num_retrieved_chunks": n_docs, "word_overlap_ratio": overlap,
            })
        except Exception as e:
            print(f"  ERROR on '{r['question'][:60]}': {e}\n")
            results.append({"chapter": r["chapter"], "question": r["question"], "error": str(e)})

    out_path = os.path.join(PROJECT_DIR, "evaluation_results_learn.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}")
    return results


def run_mark_depth_eval():
    """Feed the SAME question four answers of increasing quality and
    confirm the evaluator's score increases monotonically, for both a
    low-mark and a high-mark question."""
    test_cases = [
        {
            "question": "What is a codon?",
            "chapter": "Molecular Basis of Inheritance",
            "level": "2 Mark",
            "expected_answer": "A codon is a triplet of three nucleotides in mRNA that codes for a specific amino acid during protein synthesis.",
            "answers": {
                "wrong": "A codon is a type of protein found in the nucleus.",
                "partial": "A codon is three nucleotides.",
                "good": "A codon is a set of three nucleotides in mRNA that codes for an amino acid.",
                "complete": "A codon is a triplet of three nucleotides in mRNA. Each codon specifies a particular amino acid, and this triplet code is read during translation to synthesize proteins.",
            },
        },
        {
            "question": "Explain the process of DNA replication.",
            "chapter": "Molecular Basis of Inheritance",
            "level": "5 Mark",
            "expected_answer": "DNA replication is semiconservative: the double helix unwinds via helicase, each strand acts as a template, DNA polymerase adds complementary nucleotides in the 5' to 3' direction, the leading strand is synthesized continuously and the lagging strand discontinuously as Okazaki fragments, which are joined by DNA ligase, producing two identical DNA molecules each with one old and one new strand.",
            "answers": {
                "wrong": "DNA replication happens in the mitochondria and produces RNA.",
                "partial": "DNA replication makes a copy of DNA using an enzyme.",
                "good": "DNA unwinds and DNA polymerase copies each strand to make two new DNA molecules. It is semiconservative.",
                "complete": "DNA replication is semiconservative. Helicase unwinds the double helix, and each strand serves as a template. DNA polymerase synthesizes a new complementary strand in the 5' to 3' direction. The leading strand is made continuously, while the lagging strand is made discontinuously as Okazaki fragments, later joined by DNA ligase. The result is two DNA molecules, each with one parental and one new strand.",
            },
        },
    ]

    print("\n=== MARK-AWARE EVALUATION DEPTH TEST ===\n")
    results = []
    for case in test_cases:
        print(f"Question ({case['level']}): {case['question']}")
        row = {"question": case["question"], "level": case["level"], "scores": {}}
        for quality, answer_text in case["answers"].items():
            try:
                evaluation = evaluate_answer(
                    case["question"], answer_text, "Class 12",
                    CHAPTER_TO_FOLDER.get(case["chapter"], case["chapter"]),
                    expected_answer=case["expected_answer"], question_level=case["level"],
                )
                score = evaluation.get("score")
                row["scores"][quality] = score
                print(f"  {quality:9s} -> score {score}/10")
            except Exception as e:
                row["scores"][quality] = f"ERROR: {e}"
                print(f"  {quality:9s} -> ERROR: {e}")
        ordered = [row["scores"].get(q) for q in ("wrong", "partial", "good", "complete")]
        is_monotonic = all(
            isinstance(a, (int, float)) and isinstance(b, (int, float)) and a <= b
            for a, b in zip(ordered, ordered[1:])
        )
        row["monotonic_increase"] = is_monotonic
        print(f"  -> scores strictly non-decreasing (wrong<=partial<=good<=complete): {is_monotonic}\n")
        results.append(row)

    out_path = os.path.join(PROJECT_DIR, "evaluation_results_marks.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}")
    return results


if __name__ == "__main__":
    learn_results = run_learn_eval()
    mark_results = run_mark_depth_eval()

    print("\n=== SUMMARY ===")
    valid = [r for r in learn_results if "error" not in r]
    if valid:
        avg_overlap = sum(r["word_overlap_ratio"] for r in valid) / len(valid)
        avg_chunks = sum(r["num_retrieved_chunks"] for r in valid) / len(valid)
        print(f"Learn/RAG: {len(valid)}/{len(learn_results)} questions succeeded, "
              f"avg word-overlap groundedness proxy = {avg_overlap:.2f}, "
              f"avg retrieved chunks = {avg_chunks:.1f}")
    monotonic_count = sum(1 for r in mark_results if r.get("monotonic_increase"))
    print(f"Mark-aware evaluation: {monotonic_count}/{len(mark_results)} test cases "
          f"showed correctly increasing scores across wrong->partial->good->complete")
