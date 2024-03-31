from typing import List
import torch
from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes the sentiment analyzer with a specified model.
        """
        self.model_name = model_name
        # Use GPU if available, else CPU
        self.device = 0 if torch.cuda.is_available() else -1
        self.analyzer = pipeline("sentiment-analysis",
                                 model=self.model_name, device=self.device)

    def evaluate_sentiments(self, texts: List[str]) -> List[dict]:
        """
        Evaluates the sentiments of a list of strings.
        """
        if not texts:
            return []

        sentiments = self.analyzer(texts)
        return sentiments

    def summarize_sentiments(self, sentiments: List[dict]) -> float:
        """
        Provides a single score summarizing the overall sentiment of the analyzed texts.
        The score is calculated as the normalized difference between positive and negative sentiments.

        Args:
            sentiments (List[dict]): A list of sentiment analysis results.

        Returns:
            float: A single score representing the overall sentiment, where
                   -1.0 is entirely negative, 1.0 is entirely positive, and 0 indicates neutrality.
        """
        positive_score = sum(
            sentiment['score'] for sentiment in sentiments if sentiment['label'] == 'POSITIVE')
        negative_score = sum(
            sentiment['score'] for sentiment in sentiments if sentiment['label'] == 'NEGATIVE')
        total_score = positive_score - negative_score

        # Normalize the score based on the number of analyzed texts to get an average
        if sentiments:
            normalized_score = total_score / len(sentiments)
        else:
            normalized_score = 0  # Neutral if no texts are analyzed

        return normalized_score
