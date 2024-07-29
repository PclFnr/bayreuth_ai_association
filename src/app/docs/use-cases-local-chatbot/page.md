---
title: Building a local chatbot
nextjs:
  metadata:
    title: Building a local chatbot
    description: Learn how to build a local chat bot.
---

## Overview

In the [previous tutorial](/docs/use-cases-rag-chatbot), we have built a RAG chatbot using a closed-source LLM and embedding model from OpenAI. Since some users prefer running LLMs locally, this tutorial will demonstrate how to build a RAG chatbot using a fully local, open-source solution by changing just two Dingo components.

---

## Chatbot Architecture and Technical Stack

![Local App Architecture](https://gist.githubusercontent.com/iryna-kondr/f4779bfaa918e8af9ab1d455d63e142c/raw/4ef5627a6ce5ac37ce3ffacb786a35e49558f674/dingo_local_app_architecture.svg)

The application will consist of the following components:

1. [Streamlit](https://streamlit.io/) application: provides a frontend interface for users to interact with a chatbot.

2. FastAPI: facilitates communication between the frontend and backend.

3. [CapybaraHermes-2.5-Mistral-7B-GGUF](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF): LLM that generates responses upon receiving user queries.

4. Embedding model from [SentenceTransformers](https://sbert.net/): computes text embeddings.

5. [QDrant](https://qdrant.tech/): vector database that stores embedded chunks of text.

There are two main differences to an architecture used in the previous tutorial:

- **Usage of quantized open-source LLM:**

For running the model locally, Dingo can use [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) that is a Python binding for [`llama.cpp`](https://github.com/ggerganov/llama.cpp) library which allows to run models converted to GGUF, a binary file format for storing models for inference with `llama.cpp`.

You can find many GGUF models on [Hugging Face Hub](https://huggingface.co/models?library=gguf). We have chosen `CapybaraHermes-2.5-Mistral-7B-GGUF` model [prvided by TheBloke](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF) for this tutorial.

In order to download the model, you must go to [`Files and versions`](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/tree/main), where you will find many different files to choose from. They correspond to different [quantization types](https://huggingface.co/docs/hub/en/gguf#quantization-types) of the model. Quantization involves reducing the memory needed to store model weights by decreasing their precision (for example, from 32-bit floating points to 4-bit integers). Higher precision usually leads to a higher accuracy but also requires more computational resources, which can make the model slower and more costly to operate. Decreasing the precision allows loading large models that typically would not fit into memory, and accelerating the inference. Usually, a 4-bit quantization is considered to be an optimal balance between performance, and size/speed for LLMs.

- **Usage of open-source embedding model:**

SentenceTransformers is a Python toolkit that is built on top of Hugging Face's transformers library. It facilitates using transformer models, like BERT, RoBERTa, and others, for generating sentence embeddings. These embeddings can be used for tasks such as clustering, semantic search, and classification of texts. You can check the provided pre-trained models tuned for specific tasks either on the page of SentenceTransformers [here](https://sbert.net/docs/pretrained_models.html#model-overview), or on the [Hugging Face Hub](https://huggingface.co/models?library=sentence-transformers&sort=downloads). The models on Hugging Face Hub have a [widget](https://huggingface.co/docs/hub/models-widgets#whats-a-widget) that allows running inferences and playing with the model directly in the browser.

---

## Implementation

As the first step, we need to initialize an embedding model, a chat model and a vector store that will be populated with embedded chunks in the next step.

```python
# components.py
from agent_dingo.rag.embedders.sentence_transformer import SentenceTransformer
from agent_dingo.rag.vector_stores.qdrant import Qdrant
from agent_dingo.llm.llama_cpp import LlamaCPP

# Initialize an embedding model
embedder = SentenceTransformer(model_name="paraphrase-MiniLM-L6-v2")

# Initialize a vector store
vector_store = Qdrant(collection_name="phi_3_docs", embedding_size=384, path="./qdrant_db")

# Initialize an LLM
model = "capybarahermes-2.5-mistral-7b.Q4_K_M.gguf"
llm = LlamaCPP(model=model, n_ctx = 2048)
```

The subsequent steps involve populating the vector store, creating a RAG pipeline, and building a chatbot UI. These steps are exactly the same as in the [previous tutorial](/docs/use-cases-rag-chatbot).

By asking a question about the Phi-3 family of models, we can verify that our local model accurately retrieves the relevant information:

![Dingo Local Chatbot](https://i.ibb.co/23VmG8Y/Screenshot-2024-05-04-at-21-12-59.png)

---

## Conclusion

In this tutorial we have built a simple local chatbot that utilizes RAG technique and successfully retrieves information from a vector store to generate up-to-date responses. It can be seen that Dingo provides developers with flexibility, as the components of a LLM pipeline can be easily exchanged. For example, we were able to switch from a proprietary solution to a fully open-source solution running locally by simply changing two components of the pipeline.
