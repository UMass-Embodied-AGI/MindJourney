from openai import AzureOpenAI
from typing import Dict, List
import logging
import base64
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

logger = logging.getLogger(__name__)
class AzureConfig:
    def __init__(self, api_model, api_version="2024-12-01-preview", api_price=0.01):
        self.api_type = "azure"
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self.azure_endpoint = "YOUR_API_ENDPOINT"
        self.model = api_model
        self.limit = 30000
        self.price = api_price
        self.temperature = None
        self.top_p = None
        if self.model not in ["o4-mini", "o1"]:
            self.temperature = 0.00000001
            self.top_p = 0.0

class ChatAPI:
    def __init__(
        self,
        config: AzureConfig,
    ):
        self.messages: List[Dict[str, str]] = []
        self.history: List[Dict[str, str]] = []

        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.azure_endpoint,
        )

        self.model = config.model
        self.max_limit = min(config.limit - 2000, 20000)
        self.price = config.price
        self.usage_tokens = 0
        self.cost = 0  # USD money cost
        self.temperature = config.temperature
        self.top_p = config.top_p

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self.history.append(self.messages[-1])

    def add_user_image_message(self, image_urls, text):
        message = {
            "role": "user", 
            "content": [
                {"type": "text", "text": text}, 
            ]
        }
        for image_url in image_urls:
            base64_image = encode_image(image_url)
            message["content"].append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })
        self.messages.append(message)
        self.history.append(self.messages[-1])

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self.history.append(self.messages[-1])

    def get_system_response(self) -> str:
        return "understanding:\n\nAction: turn-right 40"
        try:
            self.do_truncation = False

            response = self.client.chat.completions.create(
                model=self.model, messages=self.messages, seed=44
            )
            response_message = response.choices[0].message

            usage_tokens = response.usage.total_tokens
            self.cost += usage_tokens * self.price / 1000
            print(
                f"[ChatGPT] current model {self.model}, usage_tokens: {usage_tokens}, "
                f"cost: ${self.cost:.5f}, price: ${self.price:.5f}"
            )
            if usage_tokens > self.max_limit:
                print(
                    f"[ChatGPT] truncate the conversation to avoid token usage limit, save money"
                )
                self.truncate()
            
            # To avoid failure response, you need to add_assistant message by yourself after you get a correct response
            # self.history.append({
            #         "role": "assistant",
            #         "content": response_message.content
            #     })
            return response_message.content
            # return "No, I am not sure. Move forward 1.575 meter."
        except Exception as e:
            logger.warning(f"[ChatGPT] Error: {e}")
            return "Sorry, I am not able to respond to that."

    def get_system_response_with_content(self, sys_prompt, content) -> str:
        try:
            self.do_truncation = False

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": content},
            ]

            if self.temperature is not None and self.top_p is not None:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature, top_p=self.top_p, seed=44
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, seed=44
                )
            
            response_message = response.choices[0].message

            usage_tokens = response.usage.total_tokens
            self.cost += usage_tokens * self.price / 1000
            print(
                f"[ChatGPT] current model {self.model}, usage_tokens: {usage_tokens}, "
                f"cost: ${self.cost:.5f}, price: ${self.price:.5f}"
            )
            if usage_tokens > self.max_limit:
                print(
                    f"[ChatGPT] truncate the conversation to avoid token usage limit, save money"
                )
                self.truncate()

            self.messages = messages
            
            # To avoid failure response, you need to add_assistant message by yourself after you get a correct response
            # self.history.append({
            #         "role": "assistant",
            #         "content": response_message.content
            #     })
            return response_message.content
            # return "No, I am not sure. Move forward 1.575 meter."
        except Exception as e:
            logger.warning(f"[ChatGPT] Error: {e}")
            return "Sorry, I am not able to respond to that."

    def get_system_response_stream(self):
        if self.temperature is not None and self.top_p is not None:
            response = self.client.chat.completions.create(
                model=self.model, messages=self.messages, temperature=self.temperature, top_p=self.top_p, stream=True, seed=44
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model, messages=self.messages, stream=True, seed=44
            )
        for chuck in response:
            if len(chuck.choices) > 0 and chuck.choices[0].finish_reason != "stop":
                if chuck.choices[0].delta.content is None:
                    continue
                yield chuck.choices[0].delta.content

        # stream mode does not support token usage check, give a rough estimation
        usage_tokens = int(sum([len(item["content"]) for item in self.message]) / 3.5)
        self.usage_tokens = usage_tokens
        self.cost += usage_tokens * self.price / 1000
        logger.info(
            f"[ChatGPT] current model {self.model}, usage_tokens approximation: {usage_tokens},"
            f" cost: ${self.cost:.2f}, price: ${self.price:.2f}"
        )

        if usage_tokens > self.max_limit:
            logger.info(
                f"[ChatGPT] truncate the conversation to avoid token usage limit"
            )
            self.truncate()

    @property
    def message(self):
        return self.messages

    @message.setter
    def message(self, message):
        """
        Usually at the dialog begining
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_first_turn},
        {"role": "assistant", "content": assistant_message_first_turn},
        """
        self.init_length = len(message)
        self.messages = message
        self.history.extend(self.messages)

    def truncate(self, percentage: int = 3):
        self.do_truncation = True
        usr_idx = [
            idx
            for idx in range(len(self.messages))
            if self.messages[idx]["role"] == "user"
        ]
        middle_idx = usr_idx[len(usr_idx) // percentage]
        logger.info(
            f"\033[33m [ChatGPT] truncate the conversation at index: {middle_idx} from {usr_idx} \033[m"
        )
        self.messages = self.messages[: self.init_length] + self.messages[middle_idx:]

    def clear(self):
        """end the conversation"""
        self.messages = []
        self.history = []



if __name__ == "__main__":
    config = AzureConfig("gpt-4o")
    chat_api = ChatAPI(config)
    chat_api.message = [
        {"role": "system", "content": "Are you chatGPT?"},
        {"role": "user", "content": "Answer Yes or No."},
    ]
    # print(chat_api.messages)
    # user_input = input("User: ")
    # chat_api.add_user_message(user_input)
    # print(chat_api.messages)
    # print(chat_api.cost)
    print(chat_api.get_system_response_with_content("You are chat assistant.", "Hello!"))
