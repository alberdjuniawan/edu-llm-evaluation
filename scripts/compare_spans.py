import argparse
import random
from pathlib import Path

from edu_eval.knowledge.diagnostics import correctness
from edu_eval.knowledge.indommlu import load_indommlu_csv
from edu_eval.knowledge.schema import KnowledgeQuestion
from edu_eval.knowledge.scorer import KnowledgeScorer
from edu_eval.models.loader import ModelLoader
from edu_eval.models.registry import ModelRegistry
from edu_eval.statistics.metrics import paired_bootstrap_ci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "IndoMMLU.csv"
MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"


def sample_questions(
    questions: list[KnowledgeQuestion], count: int, seed: int
) -> list[KnowledgeQuestion]:
    if count <= 0:
        raise ValueError("count must be positive.")

    if count > len(questions):
        raise ValueError("count exceeds available questions.")

    return random.Random(seed).sample(questions, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scoring spans.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = [
        question
        for question in load_indommlu_csv(DATA_PATH)
        if not question.is_for_fewshot
    ]
    sample = sample_questions(questions, args.count, args.seed)
    spec = ModelRegistry.from_yaml(MODELS_CONFIG).get(args.model_id)
    tokenizer, model, device = ModelLoader.load(spec)
    scored = {}

    try:
        for span in ("full", "answer"):
            scorer = KnowledgeScorer(
                model=model, tokenizer=tokenizer, device=device, span=span
            )
            scored[span] = scorer.predict_many(sample, batch_size=1)
    finally:
        del model
        del tokenizer

    for span, predictions in scored.items():
        accuracy = sum(correctness(p) for p in predictions) / len(predictions)
        print(f"{span}: n={len(predictions)} acc={accuracy:.4f}")

    full_flags = [float(correctness(p)) for p in scored["full"]]
    answer_flags = [float(correctness(p)) for p in scored["answer"]]
    low, high = paired_bootstrap_ci(full_flags, answer_flags, seed=args.seed)
    agreement = sum(a == b for a, b in zip(full_flags, answer_flags))
    print(f"agreement: {agreement}/{len(sample)}")
    print(f"paired full-answer: [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    main()
