---
title: Text summarization
nextjs:
  metadata:
    title: Text summarization
    description: Learn about text summarization.
---

## Overview

LLMs excel at performing summarization tasks. Scikit-LLM provides a summarizer that can be used both as stand-alone estimator, or as a preprocessor (in this case we can make an analogy with a dimensionality reduction preprocessor).

Example:

```python
from skllm.models.gpt.text2text.summarization import GPTSummarizer
from skllm.datasets import get_summarization_dataset

X = get_summarization_dataset()
summarizer = GPTSummarizer(model="gpt-3.5-turbo", max_words=15)
X_summarized = summarizer.fit_transform(X)
```

Please be aware that the `max_words` hyperparameter sets a soft limit, which is not strictly enforced outside of the prompt. Therefore, in some cases, the actual number of words might be slightly higher.

Additionally, it is possible to generate a summary, emphasizing a specific concept, by providing an optional parameter `focus`:

```python
summarizer = GPTSummarizer(model="gpt-3.5-turbo", max_words=15, focus="apples")
```

---

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn transformer.

### GPTSummarizer
```python
from skllm.models.gpt.text2text.summarization import GPTSummarizer
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "gpt-3.5-turbo". |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |
| `max_words`      | `int`  | Soft limit of the summary length, by default 15. |
| `focus`      | `Optional[str]`  | Concept in the text to focus on, by default None. |