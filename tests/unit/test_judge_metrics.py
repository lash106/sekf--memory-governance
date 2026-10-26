from sekf.contracts import JudgeLabel
from sekf.judge import evaluate_judge_labels, judge_label_accuracy


def test_accuracy_and_contradiction_f1() -> None:
    gold = [
        JudgeLabel.SUPPORTED,
        JudgeLabel.REFUTED,
        JudgeLabel.REFUTED,
        JudgeLabel.INSUFFICIENT_EVIDENCE,
    ]
    pred = [
        JudgeLabel.SUPPORTED,
        JudgeLabel.REFUTED,
        JudgeLabel.SUPPORTED,  # miss a contradiction
        JudgeLabel.INSUFFICIENT_EVIDENCE,
    ]
    assert judge_label_accuracy(gold, pred) == 0.75
    report = evaluate_judge_labels(gold, pred)
    assert report.n == 4
    assert report.accuracy == 0.75
    # TP=1, FN=1, FP=0 => precision=1, recall=0.5, f1=2/3
    assert abs(report.contradiction_f1 - (2.0 / 3.0)) < 1e-9
