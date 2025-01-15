---
title: Building a RAG agent
nextjs:
  metadata:
    title: Building a RAG agent
    description: Learn how to build a RAG agent.
---

## Overview

In previous tutorials, we built a pipeline that embeds the chunks of text similar to user's query to a system message, which allows the chatbot to access the external knowledge base. However, in practice, this approach may be too naive, as it:

- Embeds the data regardless its necessity;
- Does not provide a mechanism to selectively access different data sources;
- Does not allow to modify the query before retrieving the data;
- Does not allow to pass multiple queries.

All of these limitations can be addressed by building a more sophisticated pipeline logic, that might have a routing and query-rewriting mechanisms. However, a viable alternative is to use an `Agent` which can inherently perform all of these tasks.

The fundamental concept of agents involves using a language model to determine a sequence of actions (including the usage of external tools) and their order. One possible action could be retrieving data from an external knowledge base in response to a user's query. In this tutorial, we will develop a simple Agent that accesses multiple data sources and invokes data retrieval when needed.

As an example of external knowledge bases, we will use three webpages containing release announcement posts about recently released generative models:

1. [Phi-3 family of models](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/) from Microsoft;
2. [Llama 3 model](https://ai.meta.com/blog/meta-llama-3/) from Meta;
3. [OpenVoice model](https://research.myshell.ai/open-voice) from MyShell.

Since all of these models were released recently and this information was not included in GPT-4's training data, GPT can either provide no information about these topics, or may hallucinate and generate incorrect responses (see example in my previous article [here](/docs/use-cases-rag-chatbot)). By creating an agent that is able to retrieve data from external datasources (such as webpages linked above), we will provide an LLM with relevant contextual information that will be used for generating responses.

---

## RAG Agent Architecture and Technical Stack

![App Architecture](https://gist.githubusercontent.com/iryna-kondr/f4779bfaa918e8af9ab1d455d63e142c/raw/f33293fd26a27e636286b8a9285b56d120bf1cab/dingo_agent_architecture.svg)

The application will consist of the following components:

1. [Streamlit](https://streamlit.io/) application: provides a frontend interface for users to interact with a chatbot.

2. `FastAPI`: facilitates communication between the frontend and backend.

3. `Dingo Agent`: `GPT-4 Turbo` model from OpenAI that has access to provided knowledge bases and invokes data retrieval from them if needed.

4. `LLMs docs`: a vector store containing documentation about two recently released Phi-3 family of models and Llama 3.

5. `Audio gen docs`: a vector store containing documentation about recently released OpenVoice model.

6. `Embedding V3 small` model from OpenAI: computes text embeddings.

7. [QDrant](https://qdrant.tech/): vector database that stores embedded chunks of text.

---

## Implementation

### Indexing

#### Step 1:

As the first step, we need to initialize an embedding model, a chat model, and two vector stores: one for storing documentation for Llama 3 and Phi-3, and another for storing documentation for OpenVoice.

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

# Initialize a vector store with information about Phi-3 and Llama 3 models
llm_vector_store = Qdrant(collection_name="llm", embedding_size=1536, path="./qdrant_db_llm")

# Initialize a vector store with information about OpenVoice model
audio_gen_vector_store = Qdrant(collection_name="audio_gen", embedding_size=1536, path="./qdrant_db_audio_gen")

# Initialize an LLM
llm = OpenAI(model = "gpt-3.5-turbo")
```

#### Step 2:

Then, the above-mentioned websites have to be parsed, chunked into smaller pieces, and embedded. The embedded chunks are used to populate the corresponding vector stores.

```python
# build.py
from components import llm_vector_store, audio_gen_vector_store, embedder
from agent_dingo.rag.readers.web import WebpageReader
from agent_dingo.rag.chunkers.recursive import RecursiveChunker

# Read the content of the websites
reader = WebpageReader()
phi_3_docs = reader.read("https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/")
llama_3_docs = reader.read("https://ai.meta.com/blog/meta-llama-3/")
openvoice_docs = reader.read("https://research.myshell.ai/open-voice")

# Chunk the documents
chunker = RecursiveChunker(chunk_size=512)
phi_3_chunks = chunker.chunk(phi_3_docs)
llama_3_chunks = chunker.chunk(llama_3_docs)
openvoice_chunks = chunker.chunk(openvoice_docs)

# Embed the chunks
for doc in [phi_3_chunks, llama_3_chunks, openvoice_chunks]:
    embedder.embed_chunks(doc)

# Populate LLM vector store with embedded chunks about Phi-3 and Llama 3
for chunk in [phi_3_chunks, llama_3_chunks]:
    llm_vector_store.upsert_chunks(chunk)

# Populate audio gen vector store with embedded chunks about OpenVoice
audio_gen_vector_store.upsert_chunks(openvoice_chunks)
```

Run this script:

```bash
python build.py
```

#### Step 3:

Once the vector store is created, we can create a RAG pipeline. To access the pipeline from the streamlit application, we can serve it using the `serve_pipeline` function, which provides a REST API compatible with the OpenAI API (this means that we can use an official OpenAI Python client to interact with the pipeline).

```python
# serve.py
from agent_dingo.agent import Agent
from agent_dingo.serve import serve_pipeline
from components import llm_vector_store, audio_gen_vector_store, embedder, llm

agent = Agent(llm, max_function_calls=3)

# Define a function that an agent can call if needed
@agent.function
def retrieve(topic: str, query: str) -> str:
    """Retrieves the documents from the vector store based on the similarity to the query.
    This function is to be used to retrieve the additional information in order to answer users' queries.

    Parameters
    ----------
    topic : str
        The topic, can be either "large_language_models" or "audio_generation_models".
        "large_language_models" covers the documentation of Phi-3 family of models from Microsoft and Llama 3 model from Meta.
        "audio_generation_models" covers the documentation of OpenVoice voice cloning model from MyShell.
        Enum: ["large_language_models", "audio_generation_models"]
    query : str
        A string that is used for similarity search of document chunks.

    Returns
    -------
    str
        JSON-formatted string with retrieved chunks.
    """
    print(f'called retrieve with topic {topic} and query {query}')
    if topic == "large_language_models":
        vs = llm_vector_store
    elif topic == "audio_generation_models":
        vs = audio_gen_vector_store
    else:
        return "Unknown topic. The topic must be one of `large_language_models` or `audio_generation_models`"
    query_embedding = embedder.embed(query)[0]
    retrieved_chunks = vs.retrieve(k=5, query=query_embedding)
    print(f'retrieved data: {retrieved_chunks}')
    return str([chunk.content for chunk in retrieved_chunks])

# Create a pipeline
pipeline = agent.as_pipeline()

# Serve the pipeline
serve_pipeline(
    {"gpt-agent": pipeline},
    host="127.0.0.1",
    port=8000,
    is_async=False,
)
```

Run the script:

```bash
python serve.py
```

At this stage, we have an openai-compatible compatible backend with a model named `gpt-agent`, running on `http://127.0.0.1:8000/`. The Streamlit application will send requests to this backend.

#### Step 4:

Finally, we can proceed with building a chatbot UI:

```python
# app.py
import streamlit as st
from openai import OpenAI

st.title("🦊 Agent")

# provide any string as an api_key parameter
client = OpenAI(base_url="http://127.0.0.1:8000", api_key="123")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-agent"
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

🎉 We have successfully developed an agent that is augmented with the technical documentation of several newly released generative models, and can retrieve information from these documents if necessary. To assess the agent's ability to decide when to call the `retrieve` function and its effectiveness in retrieving data from external sources, we can pose some questions about the documents provided. As you can see below, the agent generated correct responses to these questions:

![Dingo Agent](https://i.ibb.co/Kh3zVGV/Screenshot-2024-05-05-at-15-33-02.png)

---

## Conclusion

In this tutorial, we have developed a RAG agent that can access external knowledge bases and retrieve data from them if needed. Unlike a "naive" RAG pipeline, the agent can selectively decide whether to access the external data, which data source to use (and how many times), and how to rewrite the user's query before retrieving the data. This approach allows the agent to provide more accurate and relevant responses, while the high-level pipeline logic remains as simple as of a "naive" RAG pipeline.
