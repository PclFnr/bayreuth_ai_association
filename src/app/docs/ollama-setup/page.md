---
title: Ollama setup
nextjs:
  metadata:
    title: Ollama setup
    description: Get started Ollama locally.
---
## 

To get started with Ollama, follow the instructions on their [download](https://ollama.com/download) page. Or if you're on Linux:


```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Configure server options as you see fit and then launch the client:
```bash
export OLLAMA_MAX_LOADED_MODELS=2 # sets the max number of loaded models
export OLLAMA_NUM_PARALLEL=2 # sets the max number of parallel tasks
ollama serve
```

Then download a model using the ollama cli:
```bash
ollama pull llama3
```
You can optionally test if everything works as you expect by using a chat session:
```bash
ollama run llama3 # runs an interactive session in the terminal
```