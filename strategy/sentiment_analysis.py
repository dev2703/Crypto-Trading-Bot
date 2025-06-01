"""
Sentiment Analysis Module
- Implements sentiment analysis using a custom LLM
- Processes financial news and social media data
- Generates sentiment scores for trading signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataset(Dataset):
    """Custom dataset for financial text data."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentAnalyzer:
    """Sentiment analyzer using a custom LLM."""
    
    def __init__(self, model_name: str = "finbert-sentiment"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text.
        
        Args:
            text: Input text to preprocess
        
        Returns:
            Preprocessed text
        """
        # Remove URLs
        text = ' '.join([word for word in text.split() if not word.startswith('http')])
        # Remove special characters
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])
        return text.strip()
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of the input text.
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary containing sentiment scores
        """
        text = self.preprocess_text(text)
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
        
        return {
            'positive': scores[0][2].item(),
            'neutral': scores[0][1].item(),
            'negative': scores[0][0].item()
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            List of sentiment score dictionaries
        """
        return [self.analyze_sentiment(text) for text in texts]
    
    def generate_signal(self, sentiment_scores: Dict[str, float]) -> int:
        """
        Generate trading signal based on sentiment scores.
        
        Args:
            sentiment_scores: Dictionary of sentiment scores
        
        Returns:
            Trading signal: 1 (buy), 0 (hold), or -1 (sell)
        """
        # Calculate sentiment score as weighted average
        score = (
            sentiment_scores['positive'] * 1 +
            sentiment_scores['neutral'] * 0 +
            sentiment_scores['negative'] * -1
        )
        
        # Generate signal based on threshold
        if score > 0.2:
            return 1  # Buy
        elif score < -0.2:
            return -1  # Sell
        else:
            return 0  # Hold
    
    def fine_tune(self, train_texts: List[str], train_labels: List[int], 
                 val_texts: List[str], val_labels: List[int],
                 epochs: int = 3, batch_size: int = 16) -> None:
        """
        Fine-tune the model on custom data.
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels
            val_texts: List of validation texts
            val_labels: List of validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Create datasets
        train_dataset = FinancialDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = FinancialDataset(val_texts, val_labels, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    val_loss += outputs.loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Training Loss: {total_loss/len(train_loader):.4f}")
            logger.info(f"Validation Loss: {val_loss/len(val_loader):.4f}")

# Example usage
if __name__ == "__main__":
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Example text
    text = "Apple Inc. reported strong quarterly earnings, exceeding market expectations."
    
    # Analyze sentiment
    sentiment_scores = analyzer.analyze_sentiment(text)
    print(f"Sentiment Scores: {sentiment_scores}")
    
    # Generate trading signal
    signal = analyzer.generate_signal(sentiment_scores)
    print(f"Trading Signal: {signal}") 