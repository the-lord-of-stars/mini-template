import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
from helpers import CUSTOM_STOP_WORDS
from state import State
from memory import shared_memory



def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def remove_stop_words(text, stop_words=None):
    if stop_words is None:
        stop_words = CUSTOM_STOP_WORDS
    
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(filtered_words)

def preprocess_documents(docs, min_df=2, max_df=0.95, remove_stop_words_flag=True):
    cleaned_docs = [clean_text(doc) for doc in docs]
    cleaned_docs = [doc for doc in cleaned_docs if doc.strip()]

    if remove_stop_words_flag:
        cleaned_docs = [remove_stop_words(doc) for doc in cleaned_docs]
        cleaned_docs = [doc for doc in cleaned_docs if doc.strip()]
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=min_df,
        max_df=max_df,
        stop_words=None,
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
    count_vectorizer = CountVectorizer(
        max_features=1000,
        min_df=min_df,
        max_df=max_df,
        stop_words=None,
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_docs)
    count_matrix = count_vectorizer.fit_transform(cleaned_docs)
    
    return cleaned_docs, tfidf_matrix, count_matrix, tfidf_vectorizer, count_vectorizer

def find_optimal_topics(count_matrix, max_topics=20):
    from sklearn.model_selection import GridSearchCV
    from sklearn.decomposition import LatentDirichletAllocation
    
    n_topics_range = list(range(2, min(max_topics + 1, count_matrix.shape[0])))
    param_grid = {
        'n_components': n_topics_range,
        'learning_decay': [0.5, 0.7, 0.9],
        'max_iter': [10]
    }
    lda = LatentDirichletAllocation(
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search = GridSearchCV(
        lda, 
        param_grid, 
        cv=3, 
        verbose=1,
        # scoring='log_likelihood'
    )
    
    grid_search.fit(count_matrix)
    
    return grid_search.best_estimator_, grid_search.best_params_
def train_lda_model(count_matrix, n_topics=10, random_state=0):

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method='batch',
        max_iter=20,
        verbose=1,
        n_jobs=-1
    )
    
    lda_model.fit(count_matrix)
    
    return lda_model

def create_pyldavis_visualization(lda_model, count_matrix, count_vectorizer, output_path=None):
    topic_term_dists = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
    doc_topic_dists = lda_model.transform(count_matrix)
    doc_lengths = count_matrix.sum(axis=1).A1
    term_frequency = count_matrix.sum(axis=0).A1
    feature_names = count_vectorizer.get_feature_names_out()

    vis_data = pyLDAvis.prepare(
            topic_term_dists=topic_term_dists,
            doc_topic_dists=doc_topic_dists,
            doc_lengths=doc_lengths,
            vocab=feature_names,
            term_frequency=term_frequency,
            mds='tsne',
            sort_topics=False
        )
    
    if output_path:
        pyLDAvis.save_html(vis_data, output_path)
    
    return vis_data