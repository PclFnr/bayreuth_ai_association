---
title: Building a RAG chatbot
nextjs:
  metadata:
    title: Building a RAG chatbot
    description: Learn how to build a RAG chat bot.
---

## Overview

Chatbots are among the most popular use cases for large language models (LLMs). They are designed to understand and respond to user inquiries, provide answers, perform tasks, or direct users to resources. Utilizing chatbots can significantly decrease customer support costs and improve response times to user requests. However, a common issue with chatbots is their tendency to deliver generic information when users expect domain-specific responses. Additionally, they may generate outdated information when users need current updates.

For demonstrations, I have chosen the webpage about [Phi-3](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/) — a family of open AI models by Microsoft released in April 2024.

If we ask how many parameters Phi-3-mini model has, GPT-4 will generate a response indicating that it does not know the answer:

```python
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How many parameters does the Phi-3-mini model from Microsoft have?"}
  ]
)

print(completion.choices[0].message)

# As of the last update, the Phi model variants by Microsoft, including the Phi-3-mini, are not explicitly defined in publicly available resources. There has been no detailed information released about a specific "Phi-3-mini" model.
```

If we ask GPT-3.5 the same question, it will hallucinate and provide incorrect information:

```python
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How many parameters does the Phi-3-mini model from Microsoft have?"}
  ]
)

print(completion.choices[0].message)

# The Phi-3-mini model from Microsoft has 121 million parameters.
```

These problems can be addressed by using the retrieval-augmented generation (RAG) technique. This technique supplements the LLM with a knowledge base external to its training data sources. For instance, an organization's internal knowledge base, such as a Wiki or internal PDFs, can be provided.

The tutorial below will demonstrate how to build a simple chatbot that utilizes RAG technique and can retrieve information about a recently released family of Phi-3 models.

---

## RAG Architecture

![RAG Architecture](https://gist.githubusercontent.com/iryna-kondr/f4779bfaa918e8af9ab1d455d63e142c/raw/ce8e33447a34db0259f888d39c58256c2cbf43b1/dingo_rag_use_case.svg)

The basic steps of the Naive RAG include:

**1. Indexing**

Indexing starts with extraction of raw data from various formats such as webpage, PDF, etc. To manage the context restrictions of language models and increase the response accuracy, the extracted text is broken down into smaller, manageable chunks. For now, Dingo supports a recursive chunking that involves breaking down a large text input into smaller segments recursively until the chunks are of a desired size. The choice of the chunking size is heavily dependent on the needs of RAG application. Thus, it is recommeded to experiment with different sizes to select the best one that will allow preserving the context and maintaining the accuracy. The extracted chunks are encoded into vector representations using an embedding model and stored in a vector database.

**2. Retrieval**

When a user submits a query, the RAG system uses the encoding model from the indexing phase to convert the query into a vector representation. It then calculates similarity scores between the query vector and the vectors of chunks in the vector database. The system identifies and retrieves the top K chunks with the highest similarity to the query. These chunks serve as the expanded context for the prompt.

**3. Generation**

The users query and selected chunks are combined into a single prompt and passed to LLM. Thus, the model is provided with the necessary contextual information to formulate and deliver a response.

---

## Chatbot Architecture and Technical Stack

![App Architecture](https://gist.githubusercontent.com/iryna-kondr/f4779bfaa918e8af9ab1d455d63e142c/raw/7f8f41d5bf00a23638b8958cc970281857a43a6f/dingo_app_architecture.svg)

On a high level, the application will consist of the following components:

1. [Streamlit](https://streamlit.io/) application: provides a frontend interface for users to interact with a chatbot.

2. `FastAPI`: facilitates communication between the frontend and backend.

3. `GPT-4 Turbo` model from OpenAI: LLM that generates responses upon receiving user queries.

4. `Embedding V3 small` model from OpenAI: computes text embeddings.

5. [QDrant](https://qdrant.tech/): vector database that stores embedded chunks of text.

---

## Implementation

### Indexing

#### Step 1:

As the first step, we need to initialize an embedding model, a chat model and a vector store that will be populated with embedded chunks in the next step.

{% callout title="Note" %}
It is needed to set OPENAI_API_KEY environment variable.
{% /callout %}

```python
# components.py
from agent_dingo.rag.embedders.openai import OpenAIEmbedder
from agent_dingo.rag.vector_stores.qdrant import Qdrant
from agent_dingo.llm.openai import OpenAI

# Initialize an embedding model
embedder = OpenAIEmbedder(model="text-embedding-3-small")

# Initialize a vector store
vector_store = Qdrant(collection_name="phi_3_docs", embedding_size=1536, path="./qdrant_db")

# Initialize an LLM
llm = OpenAI(model="gpt-4-turbo")
```

#### Step 2:

Then, the website about [Phi-3](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/) family of models has to be parsed, chunked into smaller pieces, and embedded. The embedded chunks are used to populate a vector store.

```python
# build.py
from components import vector_store, embedder
from agent_dingo.rag.readers.web import WebpageReader
from agent_dingo.rag.chunkers.recursive import RecursiveChunker

# Read the content of the website
reader = WebpageReader()
docs = reader.read("https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/")

# Chunk the document
chunker = RecursiveChunker(chunk_size=512)
chunks = chunker.chunk(docs)

# Embed the chunks
embedder.embed_chunks(chunks)

# Populate vector store with embedded chunks
vector_store.upsert_chunks(chunks)
```

Run this script:

```bash
python build.py
```

At this stage, the vector store is created, allowing chunks to be retrieved and incorporated into the prompt based on a user's query.

#### [Optional Step]

It is also possible to identify which chunks are retrieved and check their similarity scores to the user's query:

```python
# test.py
from components import vector_store, embedder
query = "How many parameters does Phi-3-mini model from Microsoft have?"
query_embedding = embedder.embed([query])[0]
# select a single chunk (k=1) with the highest similarity to the query
retrieved_chunks = vector_store.retrieve(k=1, query=query_embedding)
print(retrieved_chunks)
#[RetrievedChunk(content=' Starting today,  Phi-3-mini , a 3.8B language model is available...', document_metadata={'source': 'https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/'}, score=0.7154231207501476)]
```

We can see that the correct chunk was retrieved, which indeed contains information about the number of parameters in the Phi-3-mini model.

### Retrieval and Augmentation

#### Step 3:

Once the vector store is created, we can create a RAG pipeline and serve it.

Streamlit [only supports](https://docs.streamlit.io/develop/api-reference/chat/st.chat_message) two types of messages: `User` and `Assistant`. However, it us often more appropriate to include the retrieved data into the `System` message. Therefore, we use a custom block that injects a `System` message into the chat prompt before passing it to the RAG modifier.

```python
# serve.py
from agent_dingo.rag.prompt_modifiers import RAGPromptModifier
from agent_dingo.serve import serve_pipeline
from agent_dingo.core.blocks import InlineBlock
from agent_dingo.core.state import ChatPrompt
from agent_dingo.core.message import SystemMessage
from components import vector_store, embedder, llm

@InlineBlock()
def inject_system_message(state: ChatPrompt, context, store):
    messages = state.messages
    system_message = SystemMessage("You are a helpful assistant.")
    return ChatPrompt([system_message]+messages)

rag = RAGPromptModifier(embedder, vector_store)
pipeline = inject_system_message>>rag>>llm

serve_pipeline(
    {"gpt-rag": pipeline},
    host="127.0.0.1",
    port=8000,
    is_async=False,
)
```

Run the script:

```bash
python serve.py
```

At this stage, we have a RAG pipeline compatible with the OpenAI API, named `gpt-rag`, running on `http://127.0.0.1:8000/`. The Streamlit application will send requests to this backend.

#### Step 4:

Finally, we can proceed with building a chatbot UI:

```python
# app.py
import streamlit as st
from openai import OpenAI

st.title("🦊 LLM Expert")

# provide any string as an api_key parameter
client = OpenAI(base_url="http://127.0.0.1:8000", api_key="123")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-rag"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "🦊" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🦊"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=False,
        )
        response = st.write_stream((i for i in stream.choices[0].message.content))
    st.session_state.messages.append({"role": "assistant", "content": response})
```

Run the application:

```bash
streamlit run app.py
```

🎉 We have successfully developed a chatbot that is augmented with the technical documentation of Phi-3 family of models.
If we pose the same question to this chatbot as we previously did to GPT-4 and GPT-3.5 models, we will observe that it correctly answers the question:

![Dingo Chatbot](https://i.ibb.co/rQm0m41/Dingo-Chatbot.png)

---

## Conclusion

In this tutorial we have built a simple chatbot that utilizes RAG technique and successfully retrieves information from a vector store to generate up-to-date responses. It can be seen that Dingo enhances the development of LLM-based applications by offering essential (core) features and flexibility. That allows developers to quickly and easily create application prototypes.
