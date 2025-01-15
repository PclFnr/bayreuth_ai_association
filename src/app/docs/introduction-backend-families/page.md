---
title: Backend families
nextjs:
  metadata:
    title: Backend families
    description: Overview of backend families.
---
On a high level, Scikit-LLM estimators are divided based on the language model backend family they use. The backend family is defined by the API format and does not necessarily correspond to the language model architecture. For example, all backends that follow the OpenAI API format are groupped into _gpt_ family regardless the actual language model architecture or provider. Eeach backend family has its own set of estimators which are located in the `skllm.models.<family>` sub-module.

For example, the Zero-Shot Classifier is available as `skllm.models.gpt.classification.zero_shot.ZeroShotGPTClassifier` for the _gpt_ family, and as `skllm.models.vertex.classification.zero_shot.ZeroShotVertexClassifier` for the _vertex_ family. The separation between the backend families is necessary to allow for a reasonable level of flexibility if/when model providers introduce model-specific features that are not supported by other providers and hence cannot be easily abstracted away. At the same time, the number of model families is kept to a minimum to simplify the usage and maintenance of the library. Since the OpenAI API is by far the most popular and widely used, backends that follow that format are preferred over the others.

Whenever the backend family supports multiple backends, the default one is used unless the `model` parameter specifies a particular backend namespace. For example, the default backend for the _gpt_ family is the OpenAI backend. However, you can use the Azure backend by setting `model = "azure::<model_name>"`. However, please note that not every estimator supports every backend.

---

## GPT Family

The GPT family includes all backends that follow the OpenAI API format.


### OpenAI (default)

The OpenAI backend is the default backend for the GPT family. It is used whenever the `model` parameter does not specify a particular backend namespace.

To use the OpenAI backend, you need to set your OpenAI API key and organization ID as follows:

```python
from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("<YOUR_KEY>")
SKLLMConfig.set_openai_org("<YOUR_ORGANIZATION_ID>")
```

### Azure

OpenAI models can be alternatively used as a part of the [Azure OpenAI service](https://azure.microsoft.com/en-us/products/ai-services/openai-service). To use the Azure backend, you need to provide your Azure API key and endpoint as follows:

```python
from skllm.config import SKLLMConfig
# Found under: Resource Management (Left Sidebar) -> Keys and Endpoint -> KEY 1
SKLLMConfig.set_gpt_key("<YOUR_KEY>")
# Found under: Resource Management (Left Sidebar) -> Keys and Endpoint -> Endpoint
SKLLMConfig.set_azure_api_base("<API_BASE>") # e.g. https://<YOUR_PROJECT_NAME>.openai.azure.com/
```

When using the Azure backend, the model should be specified as `model = "azure::<model_deployment_name>"`. For example, if you created a _gpt-3.5_ deployment under the name _my-model_, you should use `model = "azure::my-model"`.

### GPT4ALL

GPT4ALL is an open-source library that provides a unified API for multiple small-scale language models, that can be run locally on a consumer-grade hardware, even without a GPU. To use the GPT4ALL backend, you need to install the corresponding extension as follows:

```bash
pip install scikit-llm[gpt4all]
```

Then, you can use the GPT4ALL by specifying the model as `model = "gpt4all::<model_name>"`, which will be downloaded automatically. For the full list of available models, please refer to the [GPT4ALL official documentation](https://gpt4all.io/index.html).

{% callout title="Note" %}
The models available through the GPT4ALL out of the box have very limited capabilities and are not recommended for most of the use cases. In addition, not all models are permitted for commercial use. Please check the license of the model you are using before deploying it in production.
{% /callout %}

### Custom URL

Custom URL backend allows to use any GPT estimator with any OpenAI-compatible provider (either running locally or in the cloud).

In order to use the backend, it is necessary to set a global custom url: 
```python
from skllm.config import SKLLMConfig

SKLLMConfig.set_gpt_url("http://localhost:8000/")

clf = ZeroShotGPTClassifier(model="custom_url::<custom_model_name>")
```


{% callout title="Note" %}
When using `custom_url` and `openai` backends within the same script, it is necessary to reset the custom url configuration using `SKLLMConfig.reset_gpt_url()`.
{% /callout %}

---

## Vertex Family

The Vertex family currently includes a single (default) backend, which is the Google Vertex AI.

In order to use the Vertex backend, you need to configure your Google Cloud credentials as follows:

1.  Log in to [Google Cloud Console](https://console.cloud.google.com/) and [create a Google Cloud project](https://developers.google.com/workspace/guides/create-project). After the project is created, select this project from a list of projects next to the Google Cloud logo (upper left corner).
2.  Search for _Vertex AI_ in the search bar and select it from the list of services.
3.  Install a Google Cloud CLI on the local machine by following [the steps from the official documentation](https://cloud.google.com/sdk/docs/install), and [set the application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials#personal) by running the following command:
    ```bash
    gcloud auth application-default login
    ```
4.  Configure Scikit-LLM with your project ID:

    ```python
    from skllm.config import SKLLMConfig

    SKLLMConfig.set_google_project("<YOUR_PROJECT_ID>")
    ```

Additionally, for tuning LLMs in Vertex, it is required to have to have 64 cores of the TPU v3 pod training resource. By default this quota is set to 0 cores and has to be increased as follows (ignore this if you are not planning to use the tunable estimators):

1.  Go to [Quotas](https://cloud.google.com/docs/quota/view-manage#requesting_higher_quota) and filter them for “Restricted image training TPU V3 pod cores per region”.
2.  Select “europe-west4” region (currently this is the only supported region).
3.  Click on “Edit Quotas”, set the limit to 64 and submit the request.
    The request should be approved within a few hours, but it might take up to several days.