---
title: Text translation
nextjs:
  metadata:
    title: Text translation
    description: Learn about text translation.
---

## Overview

LLMs have proven their proficiency in translation tasks. The actual performance will heavily depend on the efficacy of the chosen model. Nonetheless we implemented the same capabilites as `scikit-llm` and offer a locally running translation interface.

Example:

```python
from skollama.models.ollama.text2text.translation import OllamaTranslator
from skllm.datasets import get_translation_dataset

X = get_translation_dataset()
t = OllamaTranslator(model="llama3", output_language="English")
translated_text = t.fit_transform(X)
```

---

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn transformer.

### OllamaTranslator
```python
from skollama.models.ollama.text2text.translation import OllamaTranslator
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "gpt-3.5-turbo". |
| `output_language`      | `str`  | Target language, by default "English". |