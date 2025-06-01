"""
Sentiment Analysis Module
- Implements sentiment analysis using TF-IDF and Naive Bayes
- Processes financial news and social media data
- Generates sentiment scores for trading signals
"""
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    A class for analyzing sentiment in financial news and social media data.
    Uses a simple TF-IDF and Naive Bayes approach for sentiment analysis.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = MultinomialNB()
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def train(self, texts, labels):
        """
        Train the sentiment analyzer on labeled data.
        
        Args:
            texts (list): List of text samples
            labels (list): List of corresponding labels (1 for positive, 0 for neutral, -1 for negative)
        """
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Convert labels to binary (positive/negative)
            binary_labels = [1 if label > 0 else 0 for label in labels]
            
            # Vectorize texts
            X = self.vectorizer.fit_transform(processed_texts)
            
            # Train classifier
            self.classifier.fit(X, binary_labels)
            self.is_trained = True
            
            self.logger.info("Successfully trained sentiment analyzer")
            
        except Exception as e:
            self.logger.error(f"Error training sentiment analyzer: {str(e)}")
            raise
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a given text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary containing sentiment scores
        """
        try:
            if not self.is_trained:
                # If not trained, return neutral sentiment
                return {
                    'positive': 0.33,
                    'neutral': 0.34,
                    'negative': 0.33,
                    'sentiment': 0
                }
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Vectorize text
            X = self.vectorizer.transform([processed_text])
            
            # Get probability scores
            proba = self.classifier.predict_proba(X)[0]
            
            # Calculate sentiment scores
            positive_score = proba[1] if len(proba) > 1 else 0.33
            negative_score = proba[0] if len(proba) > 0 else 0.33
            neutral_score = 1 - (positive_score + negative_score)
            
            # Calculate overall sentiment (-1 to 1)
            sentiment = positive_score - negative_score
            
            return {
                'positive': positive_score,
                'neutral': neutral_score,
                'negative': negative_score,
                'sentiment': sentiment
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'positive': 0.33,
                'neutral': 0.34,
                'negative': 0.33,
                'sentiment': 0
            }
    
    def analyze_batch(self, texts):
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of sentiment analysis results
        """
        return [self.analyze_sentiment(text) for text in texts]
    
    def get_sentiment_label(self, sentiment_score):
        """
        Convert sentiment score to label.
        
        Args:
            sentiment_score (float): Sentiment score between -1 and 1
            
        Returns:
            str: Sentiment label
        """
        if sentiment_score > 0.2:
            return "Positive"
        elif sentiment_score < -0.2:
            return "Negative"
        else:
            return "Neutral"

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Example texts
    texts = [
        "The company reported strong earnings growth and positive outlook",
        "The stock market showed mixed results today",
        "The company faced significant losses and declining revenue"
    ]
    
    # Example labels (1: positive, 0: neutral, -1: negative)
    labels = [1, 0, -1]
    
    # Train the analyzer
    analyzer.train(texts, labels)
    
    # Analyze some new texts
    test_texts = [
        "The company's new product launch was successful",
        "The market remained unchanged",
        "The company's stock price dropped significantly"
    ]
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {analyzer.get_sentiment_label(result['sentiment'])}")
        print(f"Scores: {result}") 