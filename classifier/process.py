import os
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from openai import OpenAI
import requests

from config import settings
from database import Message, MessageRepository, Classification, MessageStatus

logger = logging.getLogger(__name__)


class APIRateLimitError(Exception):
    pass


def retry_with_exponential_backoff(
    func,
    initial_delay=1,
    exponential_base=2,
    jitter=True,
    max_retries=5,
    errors=(requests.exceptions.RequestException, APIRateLimitError),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    logger.error(f"Max retries exceeded: {e}")
                    raise e

                if hasattr(e, "status_code") and e.status_code == 401:
                    logger.error("Authentication error (401)")
                    raise e

                sleep_time = delay * (0.5 + random.random()) if jitter else delay
                logger.warning(
                    f"Retrying in {sleep_time:.2f}s. Attempt {num_retries}/{max_retries}"
                )
                time.sleep(sleep_time)
                delay *= exponential_base
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise e

    return wrapper


def setup_api_client() -> OpenAI:
    api_key = settings.OPENROUTER_API_KEY
    if not api_key or len(api_key) < 10:
        raise ValueError("Invalid API key")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/TeleScrape",
            "X-Title": "TeleScrape Gaza Events Classifier",
        },
    )


@retry_with_exponential_backoff
def make_api_request(client: OpenAI, model: str, messages: List[Dict]) -> Any:
    return client.chat.completions.create(
        model=model, messages=messages, extra_body={"temperature": 0.1}
    )


def parse_json_response(content: str) -> Dict[str, Any]:
    try:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_content)
        elif "```" in content:
            json_content = content.split("```")[1].split("```")[0].strip()
            return json.loads(json_content)

        if "{" in content and "}" in content:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            json_content = content[start_idx:end_idx]
            return json.loads(json_content)

        raise json.JSONDecodeError("Failed to parse JSON", content, 0)
    except Exception as e:
        logger.error(f"JSON parsing error: {e}")
        raise


PROMPT_TEMPLATE = """
أنا محلل وباحث يقوم بتوثيق الأحداث في قطاع غزة. الرجاء تحليل النصوص التالية وتحديد ما إذا كانت تشير إلى جرائم حرب أو أعمال إبادة جماعية.

معلومات مهمة للتحليل:
- إذا كان النص عبارة عن قائمة أسماء فقط (بدون سياق)، يجب اعتبار ذلك قائمة للضحايا المدنيين من عمليات القتل.
- تحديد ما إذا كان النص تصريح أو بيان رسمي من مسؤول في Gaza.

لكل نص، أجب بتنسيق JSON التالي:
{{
    "index": رقم_النص,
    "civilian_deaths": true/false,
    "targeting_civilians": true/false,
    "blocking_aid": true/false,
    "destroying_homes": true/false,
    "targeting_facilities": true/false,
    "forced_displacement": true/false,
    "systematic_violence": true/false,
    "is_official_speech": true/false,
    "is_genocidal": true/false,
    "explanation": "تفسير موجز"
}}

النصوص:
{batch_texts}

أجب بتنسيق JSON array فقط:
"""


def classify_message(
    client: OpenAI, message_text: str, model: str = None
) -> Optional[Classification]:
    if not message_text or message_text.strip() == "":
        return None

    model = model or settings.CLASSIFICATION_MODEL

    messages = [
        {
            "role": "system",
            "content": "أنت محلل متخصص في القانون الدولي الإنساني وتوثيق جرائم الحرب.",
        },
        {"role": "user", "content": PROMPT_TEMPLATE.format(message_text=message_text)},
    ]

    try:
        response = make_api_request(client, model, messages)
        content = response.choices[0].message.content
        result = parse_json_response(content)

        return Classification(
            civilian_deaths=result.get("civilian_deaths", False),
            targeting_civilians=result.get("targeting_civilians", False),
            blocking_aid=result.get("blocking_aid", False),
            destroying_homes=result.get("destroying_homes", False),
            targeting_facilities=result.get("targeting_facilities", False),
            forced_displacement=result.get("forced_displacement", False),
            systematic_violence=result.get("systematic_violence", False),
            is_official_speech=result.get("is_official_speech", False),
            is_genocidal=result.get("is_genocidal", False),
            explanation=result.get("explanation", ""),
            classified_at=datetime.utcnow(),
            model_used=model,
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return None


def classify_batch(
    client: OpenAI, messages_data: List[Dict], model: str = None
) -> Dict[int, Classification]:
    if not messages_data:
        return {}

    model = model or settings.CLASSIFICATION_MODEL

    batch_texts = []
    for idx, msg in enumerate(messages_data):
        batch_texts.append(f"{idx + 1}. {msg['text']}")

    batch_texts_str = "\n".join(batch_texts)

    messages = [
        {
            "role": "system",
            "content": "أنت محلل متخصص في القانون الدولي الإنساني وتوثيق جرائم الحرب.",
        },
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(batch_texts=batch_texts_str),
        },
    ]

    try:
        response = make_api_request(client, model, messages)
        content = response.choices[0].message.content

        results = parse_batch_response(content)

        classifications = {}
        for result in results:
            idx = result.get("index", 0) - 1
            if 0 <= idx < len(messages_data):
                msg_id = messages_data[idx]["message_id"]
                classifications[msg_id] = Classification(
                    civilian_deaths=result.get("civilian_deaths", False),
                    targeting_civilians=result.get("targeting_civilians", False),
                    blocking_aid=result.get("blocking_aid", False),
                    destroying_homes=result.get("destroying_homes", False),
                    targeting_facilities=result.get("targeting_facilities", False),
                    forced_displacement=result.get("forced_displacement", False),
                    systematic_violence=result.get("systematic_violence", False),
                    is_official_speech=result.get("is_official_speech", False),
                    is_genocidal=result.get("is_genocidal", False),
                    explanation=result.get("explanation", ""),
                    classified_at=datetime.utcnow(),
                    model_used=model,
                )

        logger.info(f"Bulk classified {len(classifications)} messages")
        return classifications

    except Exception as e:
        logger.error(f"Bulk classification error: {e}")
        return {}


def parse_batch_response(content: str) -> List[Dict[str, Any]]:
    try:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_content)
        elif "```" in content:
            json_content = content.split("```")[1].split("```")[0].strip()
            return json.loads(json_content)

        if "[" in content and "]" in content:
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            json_content = content[start_idx:end_idx]
            return json.loads(json_content)

        raise json.JSONDecodeError("Failed to parse batch JSON", content, 0)
    except Exception as e:
        logger.error(f"Batch JSON parsing error: {e}")
        raise


class MessageClassifier:
    def __init__(self):
        self.message_repo = MessageRepository()
        self.client = setup_api_client()
        self.model = settings.CLASSIFICATION_MODEL
        self.bulk_size = 10

    def process_batch(self, limit: int = None) -> int:
        limit = limit or settings.BATCH_SIZE

        messages = self.message_repo.get_messages_by_status(MessageStatus.NEW, limit)

        if not messages:
            logger.info("No new messages to classify")
            return 0

        logger.info(f"Processing {len(messages)} messages in bulk")

        processed = 0

        for i in range(0, len(messages), self.bulk_size):
            batch = messages[i : i + self.bulk_size]

            messages_data = [
                {"message_id": msg.message_id, "text": msg.text, "channel": msg.channel}
                for msg in batch
            ]

            classifications = classify_batch(self.client, messages_data, self.model)

            for msg in batch:
                classification = classifications.get(msg.message_id)
                if classification:
                    success = self.message_repo.update_classification(
                        msg.message_id, msg.channel, classification
                    )
                    if success:
                        processed += 1
                        logger.debug(f"Classified message {msg.message_id}")

            time.sleep(2)

        logger.info(f"Classified {processed}/{len(messages)} messages")
        return processed


def main():
    import random

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    classifier = MessageClassifier()
    count = classifier.process_batch()
    print(f"Classified {count} messages")


if __name__ == "__main__":
    main()
