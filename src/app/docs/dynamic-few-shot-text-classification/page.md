---
title: Dynamic few-shot text classification
nextjs:
  metadata:
    title: Dynamic few-shot text classification
    description: Learn about dynamic few-shot text classification.
---

## Overview

Dynamic Few-Shot Classification is an extension of [Few-Shot Text Classification](/docs/few-shot-text-classification) that is more suitable for larger datasets. Instead of using a fixed set of examples for each class, it constructs a dynamic subset for each sample on the fly. This allows to efficiently utilize the limited contex window of the model and save the number of consumed tokens.

Dynamic Few-Shot Classification can be motivated from a variety of ways, from prior studies show-casing how kNN-Pretraining improves question answering capabilities ([[1]](https://arxiv.org/pdf/2110.04541)), or literature that shows, how dynamically adjusting the prompt template can improve performance ([[2]](https://arxiv.org/pdf/2108.13161)). In this sense it would also make sense, that dynamically adjusting the examples may be better for performance as well. However, hand-crafting each prompt is an exhaustive task ([[3]](https://arxiv.org/pdf/2104.08786)) and not in the spirit of automated classification.

Let's consider a toy example, where the goal is to determine whether the review is about a book or a movie. The training dataset consists of 6 samples, 3 for each class:

```python
X = [
    "I love reading science fiction novels, they transport me to other worlds.", # example 1 - book - sci-fi
    "A good mystery novel keeps me guessing until the very end.", # example 2 - book - mystery
    "Historical novels give me a sense of different times and places.", # example 3 - book - historical
    "I love watching science fiction movies, they transport me to other galaxies.", # example 4 - movie - sci-fi
    "A good mystery movie keeps me on the edge of my seat.", # example 5 - movie - mystery
    "Historical movies offer a glimpse into the past.", # example 6 - movie - historical
]

y = ["books", "books", "books", "movies", "movies", "movies"]
```

Now let's say we want to classify the following review:

```bash
I have fallen deeply in love with this sci-fi book; its unique blend of science and fiction has me spellbound.
```

Since the query is about a sci-fi book, we would like to only examples 1 and 4 to be used for classification, since they are the most relevant. If we use the dynamic few-shot classifier with 1 example per class, and investigate which examples were selected, we can see that the model successfully identified examples 1 and 4 as the most relevant ones:

```python
from skollama.models.ollama.classification.few_shot import DynamicFewShotOllamaClassifier

query = "I have fallen deeply in love with this sci-fi book; its unique blend of science and fiction has me spellbound."

clf = DynamicFewShotOllamaClassifier(n_examples=1).fit(X,y)

prompt = clf._get_prompt(query)
print(prompt)
```

```bash
...

Sample input:
"I love reading science fiction novels, they transport me to other worlds."

Sample target: books


Sample input:
"I love watching science fiction movies, they transport me to other galaxies."

Sample target: movies

...
```

This is achieved by adding a KNN search algorithm as an additional preprocessor. If we assume that the most relevant examples are the closest ones in space, then the problem reduces to a nearest neighbors search and can be tackled in three steps:

1. **Vectorization:**
   Before doing the nearest neighbors search, the training set must be vectorized using an embedding model. By default `Scikit-Ollama` uses `nomic-embed-text` for embedding and `llama3` for chat completion.
2. **Index construction** using an arbitrary nearest neighbors search algorithm. The index allows to efficiently retrieve the nearest neighbors from each class. Currently [Scikit-Learn KNN](https://scikit-learn.org/stable/modules/neighbors.html) and [Annoy](https://github.com/spotify/annoy) are supported, but it is possible to add a custom index as well.
3. **Balanced sampling:**
   The last thing to be accounted for is a class balancing. If only N nearest neighbors are selected for a few-shot prompting, there is a very high risk that some of the classes will be underrepresented or missing completely. To mitigate this issue, instead of creating a single index, the training data is partitioned by class. In this way, we are able to sample N examples from each class, ensuring the equal representation of each class.

---

## API Reference

The following API reference only lists the parameters needed for the initialization of the estimator. The remaining methods follow the syntax of a scikit-learn classifier.

### DynamicFewShotOllamaClassifier
```python
from skollama.models.ollama.classification.few_shot import DynamicFewShotOllamaClassifier
```

| **Parameter** | **Type** | **Description**          |
| ------------- | -------- | ------------------------ |
| `model`      | `str`  | Model to use, by default "gpt-3.5-turbo". |
| `default_label`      | `str`  | Default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random". |
| `prompt_template`      | `Optional[str]`  | Custom prompt template to use, by default None. |
| `key`      | `Optional[str]`  | Estimator-specific API key; if None, retrieved from the global config, by default None. |
| `n_examples`      | `int`  | Number of closest examples per class to be retrieved, by default 3. |
| `memory_index`      | `Optional[IndexConstructor]`  | Custom memory index, for details check `skllm.memory` submodule, by default None. |
| `vectorizer`      | `Optional[BaseVectorizer]`  | Scikit-LLM vectorizer; if None, `GPTVectorizer` is used, by default None. |
| `metric`  | `Optional[str]` | Metric used for similarity search by the memory_index, by default "euclidean" 