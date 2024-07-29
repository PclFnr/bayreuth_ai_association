---
title: Quick start
nextjs:
  metadata:
    title: Quick start
    description: Get started with Dingo.
---

## 

To get started with Dingo, you can install the framework using pip:

```bash
pip install agent-dingo
```

Now we can create a simple pipeline that summarizes the text provided as input and translates it into French. In this particular example, we will use GPT-3.5 model from OpenAI, but Dingo supports other LLM providers as well.

Firstly, make sure to set the `OPENAI_API_KEY` environment variable to your OpenAI API key:

```bash
export OPENAI_API_KEY=your-api-key
```

Next, create a new Python script and import the necessary modules:

```python
from agent_dingo.llm.openai import OpenAI
from agent_dingo.core.blocks import PromptBuilder
from agent_dingo.core.message import UserMessage
from agent_dingo.core.state import ChatPrompt
```

Then, define the pipeline by creating and chaining the building blocks together:

````python
# Model
gpt = OpenAI("gpt-3.5-turbo")

# Summary prompt block
summary_pb = PromptBuilder(
    [UserMessage("Summarize the text in 10 words: ```{text}```.")]
)

# Translation prompt block
translation_pb = PromptBuilder(
    [UserMessage("Translate the text into {language}: ```{summarized_text}```.")],
    from_state=["summarized_text"],
)

# Pipeline
pipeline = summary_pb >> gpt >> translation_pb >> gpt
````

Finally, run the pipeline with the input text:

```python
input_text = """
Dingo is an ancient lineage of dog found in Australia, exhibiting a lean and sturdy physique adapted for speed and endurance, dingoes feature a wedge-shaped skull and come in colorations like light ginger, black and tan, or creamy white. They share a close genetic relationship with the New Guinea singing dog, diverging early from the domestic dog lineage. Dingoes typically form packs composed of a mated pair and their offspring, indicating social structures that have persisted through their history, dating back approximately 3,500 years in Australia.
"""

output = pipeline.run(text = input_text, language = "french")
print(output)
```

To deploy the pipeline as a web service, you can use the following code:

```python
#server.py
from agent_dingo.serve import serve_pipeline

serve_pipeline({"gpt-summary-translation": pipeline}, port=8000)
```

This will start a web server on port 8000, exposing the pipeline as a REST API. You can now send requests using any HTTP client, or even using the official OpenAI python client library:

```python
# client.py
import openai

client = openai.OpenAI(base_url = "http://localhost:8000")

messages = [
    {"role": "context_text", "content": "<text to summarize>"},
    {"role": "context_language", "content": "french"},
]

out = client.chat.completions.create(messages = messages, model = "gpt-summary-translation")
print(out)
```

In this example, we have created a simple pipeline which is not designed for multi-turn conversations. To make it compatible with OpenAI Chat structure, Dingo defines special message roles like `context_text` and `context_language` which are used to pass the input arguments to the pipeline. Section [Core](/docs/core-overview) goes into more details on differences between context and chat inputs and how to handle them in Dingo.