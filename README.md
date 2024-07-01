<div align="center">
    <h1 align="center">Set up Multi-LLM Routing with BentoML</h1>
</div>

This is a BentoML example project, showing you how to serve and deploy a multi-LLM app. Before generating a response, the app assesses whether the prompt contains toxic content. If it is considered toxic, the server will not produce a corresponding response. If the prompt is non-toxic, the app will route requests to the specified LLM.

[Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [GPT-3.5 Turbo, and GPT-4o](https://platform.openai.com/docs/models) are included in this project. You can integrate other LLMs by refering to the [BentoVLLM example project](https://github.com/bentoml/BentoVLLM).

See [here](https://github.com/bentoml/BentoML/tree/main/examples) for a full list of BentoML example projects.

## Prerequisites

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- The project uses the model `mistralai/Mistral-7B-Instruct-v0.2`, which [requires you to accept relevant conditions to gain access](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2). 
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Install dependencies

```bash
# Clone the repo
git clone https://github.com/bentoml/llm-router.git
cd llm-router
pip install -r requirements.txt

# Set your OpenAI key env var
export OPENAI_API_KEY=XXXXX
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .

2024-07-01T12:44:30+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:LLMRouter" listening on http://localhost:3000 (Press CTRL+C to quit)
```


The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -s -X POST \
    'http://localhost:3000/generate' \
    -H 'Content-Type: application/json' \
    -d '{
        "max_tokens": 1024,
        "model": "mistral", # You can also set "gpt-3.5-turbo" or "gpt-4o"
        "prompt": "Explain superconductors like I'\''m five years old"
    }' 
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response_generator = client.generate(
        max_tokens=1024,
        model="mistral", # You can also set "gpt-3.5-turbo" or "gpt-4o"
        prompt="Explain superconductors like I'm five years old",
    )
    for response in response_generator:
        print(response, end='')
```

</details>

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), set the environment variables for Hugging Face and OpenAI in `bentofile.yaml`, then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
