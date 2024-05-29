import bentoml
from bentoml.exceptions import BentoMLException, BadInput
from openai import AsyncOpenAI

from mistral import MistralService

from annotated_types import Ge, Le
from typing_extensions import Annotated
from typing import AsyncGenerator


MAX_TOKENS = 1024

@bentoml.service
class LLMRouter:

    mistral = bentoml.depends(MistralService)
    # classifier = bentoml.depends(BertService)

    def __init__(self):
        self.openai_client = AsyncOpenAI()

    @bentoml.api
    async def generate_mistral(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        # query_type = (await classifier.classify(prompt))[0]['label']

        # if query_type == "toxic":
        #     raise BadInput("Request Refused..")
        #
        # if query_type == "require_db":
        #     prompt = rag_engine.enrich(prompt)
        #     return self.mistral
        #     
        # if query_type == ""
        return self.mistral.generate(prompt, max_tokens)

    
    @bentoml.api
    async def generate_openai(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        res = await self.openai_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            model="gpt-3.5-turbo",
            max_tokens=max_tokens,
            stream=True
        )
        async for chunk in res:
            yield chunk.choices[0].delta.content or ""






