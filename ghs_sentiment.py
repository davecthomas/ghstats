import os
from typing import List
import pandas as pd
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


class GhsSentimentAnalyzer:
    TRAINING_DIR = "training"
    PERCENTAGE_SAMPLE_SIZE = 1 / 3
    MODEL_MAX_TOKEN_LENGTH = 512        # Longer comments will be truncated
    MAX_COMMENT_LENGTH = 1000           # Comments longer than this will be truncated

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes the sentiment analyzer with a specified model.
        """
        self.model_name = model_name
        # Use GPU if available, else CPU
        self.device = 0 if torch.cuda.is_available() else -1
        # Initialize the tokenizer used to preprocess the text to truncate longer comments
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Initialize the sentiment analysis pipeline.
        self.analyzer = pipeline("sentiment-analysis",
                                 model=self.model_name, device=self.device)

    def count_tokens(self, texts: List[str]) -> List[int]:
        token_counts = [len(self.tokenizer.tokenize(text)) for text in texts]
        return token_counts

    def truncate_comments(self, texts: List[str]) -> List[str]:
        truncated_texts = []
        count_truncated_comments = 0
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            # Check if truncation is necessary
            if len(tokens) > self.MODEL_MAX_TOKEN_LENGTH:
                # Truncate the tokens and add the separator token
                truncated_tokens = tokens[:self.MODEL_MAX_TOKEN_LENGTH -
                                          1] + [self.tokenizer.sep_token]
                # Convert the truncated tokens back to string
                truncated_text = self.tokenizer.convert_tokens_to_string(
                    truncated_tokens)
                truncated_texts.append(truncated_text)
                count_truncated_comments += 1
            else:
                # For texts within the limit, append the original text
                truncated_texts.append(text)
            print("Truncated comments that were too long for the model: ",
                  count_truncated_comments)
        return truncated_texts

    def evaluate_sentiments(self, texts: List[str]) -> List[dict]:
        """
        Evaluates the sentiments of a list of strings.
        """
        if not texts:
            return []

        sentiments = self.analyzer(texts)
        return sentiments

    def evaluate_sentiments(self, comments: List[str]) -> List[dict]:
        if not comments:
            return []

        sentiments = self.analyzer(comments)
        return sentiments

    def output_to_training_csv(self, comments: List[str], sentiments: List[dict], filename: str = "training_dataset.csv"):

        # Prepare data for the CSV
        data = []
        for text, sentiment in zip(comments, sentiments):
            # Mapping sentiment labels to numerical labels if needed
            # Note that the default model doesn't have a NEUTRAL label, so there won't be any stored with a label of 0
            # A human trainer would be expected to add these manually
            label = {"NEGATIVE": -1, "NEUTRAL": 0,
                     "POSITIVE": 1}[sentiment['label']]
            data.append({"text": text, "label": label})

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Randomly select 1/3 of the entries to make sure we get a good sampling since the linear
        # order of the code review comments may affect the training
        df_sampled = df.sample(
            frac=self.PERCENTAGE_SAMPLE_SIZE) if len(df) > (1/self.PERCENTAGE_SAMPLE_SIZE)-1 else df

        # Ensure the subdirectory exists
        os.makedirs(self.TRAINING_DIR, exist_ok=True)

        # Save to CSV
        csv_path = os.path.join(self.TRAINING_DIR, filename)
        df_sampled.to_csv(csv_path, index=False)

    def summarize_sentiments(self, sentiments: List[dict]) -> List[float]:
        """
        Provides a single score summarizing the overall sentiment of the analyzed texts.
        The score is calculated as the normalized difference between positive and negative sentiments.

        Args:
            sentiments (List[dict]): A list of sentiment analysis results.

        Returns:
            a list of positive score, negative score, total score, and normalized score (an average of total score/len(sentiments))
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

        return [positive_score, negative_score, total_score, normalized_score]
