---
title: Tunable text-to-text
nextjs:
  metadata:
    title: Tunable text-to-text
    description: Learn about tuning for text-to-text tasks.
---

## Overview

Tunable text-to-text estimators are estimators that can be tuned to perform a variety of tasks, including but not limited to text summarization, question answering, and text translation. These estimators use the provided data as-is, without any additional preprocessing, or constructing prompts. While this approach allows for more flexibility, it is the user's responsibility to ensure that the data is formatted correctly.

```python
from skllm.models.gpt.text2text.tunable import TunableGPTText2Text

model = TunableGPTText2Text(
        base_model = "gpt-3.5-turbo-0613",
        n_epochs = None, # int or None. When None, will be determined automatically by OpenAI
        system_msg = "You are a text processing model."
)

model.fit(X_train, y_train) # y_train is any desired output text
out = model.transform(X_test)
```

---

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn transformer.

### TunableGPTText2Text
```python
from skllm.models.gpt.text2text.tunable import TunableGPTText2Text
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `base_model`      | `str`  | Model to use, by default "gpt-3.5-turbo-0613". |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `org`      | `Optional[str]`  | Estimator-specific ORG key; if None, retrieved from the global config, by default None. |
| `n_epochs`      | `Optional[int]`  | Number of epochs; if None, determined automatically; by default None. |
| `custom_suffix`      | `Optional[str]`  | Custom suffix of the tuned model, used for naming purposes only, by default "skllm". |

### TunableVertexText2Text
```python
from skllm.models.vertex.text2text.tunable import TunableVertexText2Text
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `base_model`      | `str`  | Model to use, by default "text-bison@002". |
| `n_update_steps`      | `int`  | Number of epochs, by default 1. |