---
title: Zero-shot text classification
nextjs:
  metadata:
    title: Zero-shot text classification
    description: Learn about zero-shot text classification.
---

## Overview

One of the powerful features of LLMs is the ability to perform text classification without being re-trained. For that, the only requirement is that the labels must be descriptive.

For example, let's consider a task of classifying a text into one of the following categories: [positive, negative, neutral]. We will use a class `ZeroShotGPTClassifier` and a regular scikit-learn API to perform the classification:

```python
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
from skllm.datasets import get_classification_dataset

# demo sentiment analysis dataset
# labels: positive, negative, neutral
X, y = get_classification_dataset()

clf = ZeroShotGPTClassifier(model="gpt-3.5-turbo")
clf.fit(X,y)
labels = clf.predict(X)
```

However, in the zero-shot setting, the training data is not required as it is only used for the extraction of the candidate labels. Therefore, it is sufficient to manually provide a list of candidate labels:

```python
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
from skllm.datasets import get_classification_dataset

X, _ = get_classification_dataset()

clf = ZeroShotGPTClassifier()
clf.fit(None, ["positive", "negative", "neutral"])
labels = clf.predict(X)
```

Additionally, it is possible to perform the classification in a multi-label setting, where multiple labels can be assigned to a single text at a same time:

```python
from skllm.models.gpt.classification.zero_shot import MultiLabelZeroShotGPTClassifier
from skllm.datasets import get_multilabel_classification_dataset

X, _ = get_multilabel_classification_dataset()
candidate_labels = [
    "Quality",
    "Price",
    "Delivery",
    "Service",
    "Product Variety",
    "Customer Support",
    "Packaging",
    "User Experience",
    "Return Policy",
    "Product Information",
]
clf = MultiLabelZeroShotGPTClassifier(max_labels=3)
clf.fit(None, [candidate_labels])
labels = clf.predict(X)
```

{% callout title="Note" %}
Unlike in a typical supervised setting, the performance of a zero-shot classifier greatly depends on how the label itself is structured. It has to be expressed in natural language, be descriptive and self-explanatory. For example, in the previous semantic classification task, it could be beneficial to transform a label from `<<SEMANTICS>>` to `the semantics of the provided text is <<SEMANTICS>>`.
{% /callout %}

---

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn classifier.

### ZeroShotGPTClassifier

```python
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "gpt-3.5-turbo". |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |
| `prompt_template`      | `Optional[str]`  | Custom prompt template to use, by default None. |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |

### MultiLabelZeroShotGPTClassifier

```python
from skllm.models.gpt.classification.zero_shot import MultiLabelZeroShotGPTClassifier
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "gpt-3.5-turbo". |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |
| `max_labels`      | `Optional[int]`  | Maximum labels per sample, by default 5. |
| `prompt_template`      | `Optional[str]`  | Custom prompt template to use, by default None. |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |

### ZeroShotVertexClassifier

```python
from skllm.models.vertex.classification.zero_shot import ZeroShotVertexClassifier
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "text-bison@002". |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |
| `prompt_template`      | `Optional[str]`  | Custom prompt template to use, by default None. |

### MultiLabelZeroShotVertexClassifier

```python
from skllm.models.vertex.classification.zero_shot import MultiLabelZeroShotVertexClassifier
```
| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "text-bison@002". |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |
| `max_labels`      | `Optional[int]`  | Maximum labels per sample, by default 5. |
| `prompt_template`      | `Optional[str]`  | Custom prompt template to use, by default None. |