---
title: Text vectorization
nextjs:
  metadata:
    title: Text vectorization
    description: Learn about text vectorization.
---

## Overview

LLMs can be used solely for data preprocessing by embedding a chunk of text of arbitrary length to a fixed-dimensional vector, that can be further used with virtually any model (e.g. classification, regression, clustering, etc.).

With Scikit-Ollama you can choose from a large variety of embedding models. The quality of which you can check on leaderboards such as Huggingface's [MTEB](https://huggingface.co/spaces/mteb/leaderboard). In the following example we will work with the default `nomic-embed-text` embedding model. Simply download it using the usual Ollama CLI command:
```bash
ollama pull nomic-embed-text
```
Example 1: Embedding the text

```python
from skollama.models.ollama.vectorization import OllamaVectorizer

vectorizer = OllamaVectorizer(batch_size=2) # batch_size is number of parallel tasks
X = vectorizer.fit_transform(["This is a text", "This is another text"])
```

Example 2: Combining the vectorizer with the XGBoost classifier in a scikit-learn pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

steps = [("Ollama", OllamaVectorizer()), ("Clf", XGBClassifier())]
clf = Pipeline(steps)
clf.fit(X_train, y_train_encoded)
yh = clf.predict(X_test)
```

---

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn transformer.

### OllamaVectorizer
```python
from skllm.models.gpt.vectorization import OllamaVectorizer
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "text-embedding-3-small". |
| `batch_size`      | `int`  | Number of samples per request, by default 1. |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |