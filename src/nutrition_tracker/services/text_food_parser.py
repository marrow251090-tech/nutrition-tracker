
"""Parse ordinary Russian food messages into structured food items."""

import json
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ParsedFoodItem:
    """A food item extracted from a user's message."""

    name: str
    quantity: float
    unit: str
    preparation: str | None = None


@dataclass(frozen=True)
class ParsedFoodMessage:
    """Result of parsing a food message."""

    items: list[ParsedFoodItem]
    needs_clarification: bool = False
    clarification_question: str | None = None


class TextFoodParser:
    """Extract foods and quantities from Russian text using an OpenAI-compatible API."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        reasoning_effort: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort

    async def parse(self, text: str) -> ParsedFoodMessage:
        """Parse a message without calculating nutrition or saving it."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit": {
                                "type": "string",
                                "enum": ["g", "ml", "piece", "serving"],
                            },
                            "preparation": {
                                "type": ["string", "null"],
                            },
                        },
                        "required": [
                            "name",
                            "quantity",
                            "unit",
                            "preparation",
                        ],
                        "additionalProperties": False,
                    },
                },
                "needs_clarification": {"type": "boolean"},
                "clarification_question": {
                    "type": ["string", "null"],
                },
            },
            "required": [
                "items",
                "needs_clarification",
                "clarification_question",
            ],
            "additionalProperties": False,
        }

        instructions = """
Ты разбираешь сообщения пользователя о съеденной еде.
Верни только JSON, соответствующий заданной схеме.

Правила:
- Понимай русский язык, разговорные фразы и опечатки.
- Выделяй каждый продукт отдельно.
- Сохраняй указанное количество.
- Единицы: g для граммов, ml для миллилитров,
  piece для штук, serving для порций.
- Переводи килограммы в граммы.
- Для яиц и других продуктов, указанных в штуках,
  используй piece.
- В name указывай конкретный продукт.
- В preparation указывай способ приготовления, если он известен:
  например, варёное, жареное, запечённое, варёная гречка.
- Не придумывай вес, которого пользователь не указал.
- Не считай калории и БЖУ.
- Если количество не указано и его нельзя надёжно определить,
  запроси уточнение.
- Если у гречки, риса, макарон или другого продукта неясно,
  указан вес сухого или готового продукта, запроси уточнение,
  когда это существенно для расчёта.
- Если сообщение не содержит описания еды, верни пустой items
  и запроси уточнение.
- Не считай упоминание продукта доказательством того,
  что пользователь его съел: ориентируйся на смысл сообщения.
"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        instructions
                        + "\n\nJSON schema:\n"
                        + json.dumps(schema, ensure_ascii=False)
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            response_format={"type": "json_object"},
        )

        output_text = response.choices[0].message.content

        if not output_text:
            raise RuntimeError("ClaudePoint returned an empty response")

        data = json.loads(output_text)

        items = [
            ParsedFoodItem(
                name=str(item["name"]).strip(),
                quantity=float(item["quantity"]),
                unit=str(item["unit"]),
                preparation=item.get("preparation"),
            )
            for item in data["items"]
            if str(item["name"]).strip() and float(item["quantity"]) > 0
        ]

        return ParsedFoodMessage(
            items=items,
            needs_clarification=bool(data["needs_clarification"]),
            clarification_question=data.get("clarification_question"),
        )
