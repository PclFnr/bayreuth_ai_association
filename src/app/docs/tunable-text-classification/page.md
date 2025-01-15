---
title: Tunable text classification
nextjs:
  metadata:
    title: Tunable text classification
    description: Learn about tuning for text classification tasks.
---

## Overview

Tunable estimators allow to fine-tune the underlying LLM for a classification task. Usually, tuning is performed directly in the cloud (e.g. OpenAI, Vertex), therefore it is not required to have a GPU on your local machine. However, be aware that tuning can be costly and time-consuming. We recommend to first try the in-context learning estimators, and only if they do not provide satisfactory results, to try the tunable estimators.

Example using GPT-3.5-Turbo-0613:

```python
from skllm.models.gpt.classification.tunable import GPTClassifier

X, y = get_classification_dataset()
clf = GPTClassifier(n_epochs=1)
clf.fit(X,y)
clf.predict(X)
```

---

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn classifier.

### GPTClassifier
```python
from skllm.models.gpt.classification.tunable import GPTClassifier
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `base_model`      | `str`  | Base model to use, by default "gpt-3.5-turbo-0613". |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |
| `n_epochs`      | `Optional[int]`  | Number of epochs; if None, determined automatically; by default None. |
| `custom_suffix`      | `Optional[str]`  | Custom suffix of the tuned model, used for naming purposes only, by default "skllm". |
| `prompt_template`      | `Optional[str]`  | Custom prompt template to use, by default None. |

### MultiLabelGPTClassifier
```python
from skllm.models.gpt.classification.tunable import MultiLabelGPTClassifier
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `base_model`      | `str`  | Base model to use, by default "gpt-3.5-turbo-0613". |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |
| `n_epochs`      | `Optional[int]`  | Number of epochs; if None, determined automatically; by default None. |
| `custom_suffix`      | `Optional[str]`  | Custom suffix of the tuned model, used for naming purposes only, by default "skllm". |
| `prompt_template`      | `Optional[str]`  | Custom prompt template to use, by default None. |
| `max_labels`      | `Optional[int]`  | Maximum labels per sample, by default 5. |

### VertexClassifier
```python
from skllm.models.vertex.classification.tunable import VertexClassifier
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `base_model`      | `str`  | Base model to use, by default "text-bison@002". |
| `n_update_steps`      | `int`  | Number of epochs, by default 1. |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |