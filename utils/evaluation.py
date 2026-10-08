from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def calculate_scores(reference, generated):
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()

    bleu = sentence_bleu(reference_tokens, generated_tokens)
    meteor = meteor_score([reference.split()], generated.split())

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(reference, generated)

    return {
        "BLEU": round(bleu, 3),
        "METEOR": round(meteor, 3),
        "ROUGE-1": round(rouge['rouge1'].fmeasure, 3),
        "ROUGE-L": round(rouge['rougeL'].fmeasure, 3)
    }