import bentoml
from bentoml.exceptions import BentoMLException, BadInput
from openai import AsyncOpenAI

from mistral import MistralService
from toxic_detect import ToxicClassifier

from annotated_types import Ge, Le
from typing_extensions import Annotated
from typing import AsyncGenerator
from enum import Enum, IntEnum
from pydantic import BaseModel, ValidationError


class ModelName(str, Enum):
    gpt3 = 'gpt3'
    mistral = 'mistral'

MAX_TOKENS = 1024

@bentoml.service
class LLMRouter:

    mistral = bentoml.depends(MistralService)
    toxic_classifier = bentoml.depends(ToxicClassifier)

    def __init__(self):
        self.openai_client = AsyncOpenAI()


    @bentoml.api
    async def generate_mistral(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:

        response = self.mistral.generate(prompt, max_tokens)
        async for chunk in response:
            yield chunk

    
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



    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        model: ModelName = "mistral",
    ) -> AsyncGenerator[str, None]:
        res = self.toxic_classifier.classify([prompt])[0]['label']

        if res is "toxic":
            raise BadInput("Toxic input detected")
        else:
            if model == "mistral":
                gen = self.generate_mistral(prompt)
            else:
                gen = self.generate_openai(prompt)

            async for chunk in gen:
                yield chunk
