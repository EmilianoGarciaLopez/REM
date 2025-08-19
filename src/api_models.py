import base64
import json
import logging
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from functools import partial

import anthropic
import google.generativeai as genai
import tiktoken
from openai import APIConnectionError, OpenAI, RateLimitError
from PIL import Image
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from term_image.image import from_file
from termcolor import colored

# Load environment variables from a .env file if present (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")


class Conversation(ABC):
    @abstractmethod
    def __init__(self, user_prompt, system_prompt=None):
        pass

    @abstractmethod
    def add_message(self, message):
        pass


class OpenAIConversation(Conversation):
    def __init__(self, user_prompt, system_prompt=None):
        self.messages = []
        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})
        first_message = {"role": "user", "content": user_prompt}
        self.messages.append(first_message)

    def add_message(self, message):
        self.messages.append(message)


class GeminiConversation(Conversation):
    def __init__(self, user_prompt, system_prompt=None):
        self.messages = []
        if system_prompt is not None:
            self.messages.append({"role": "user", "parts": system_prompt})
            self.messages.append({"role": "model", "parts": "Understood."})
        first_message = {"role": "user", "parts": user_prompt}
        self.messages.append(first_message)

    def add_message(self, message):
        assert type(message["parts"]) == list, "Message must be in Gemini format"
        self.messages.append(message)


class AnthropicConversation(Conversation):
    def __init__(self, user_prompt, system_prompt=None):
        # Anthropic system prompt goes in API call
        self.messages = []
        first_message = {"role": "user", "content": user_prompt}
        self.messages.append(first_message)

    def add_message(self, message):
        self.messages.append(message)


class OLlamaConversation(Conversation):
    def __init__(self, user_prompt, system_prompt=None):
        self.messages = []
        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})
        self.add_message(user_prompt)

    def add_message(self, message):
        self.messages.append(message)


class Model(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def call_model(self, user_prompt, system_prompt=None, image_paths=None):
        pass

    def count_tokens(
        self, system_prompt, user_prompt, assistant_response, image_paths=None
    ):
        pass

    def encode_image(self, image_path):
        # goes from image filepath to base64 encoding needed for APIs
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def resize_image(self, image_path, max_size_mb=5, quality=85):
        """Resize image to a temp file using JPG format for better compression."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB mode if necessary (in case of RGBA images)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Resize image
                img = img.resize((960, 640), Image.LANCZOS)

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    temp_path = tmp.name
                    # Save as JPG with initial quality
                    img.save(temp_path, format="JPEG", optimize=True, quality=quality)

                    file_size = os.path.getsize(temp_path)
                    max_size_bytes = max_size_mb * 1024 * 1024

                    # Reduce quality if file is too large
                    while file_size > max_size_bytes and quality > 10:
                        quality = int(quality * 0.9)
                        img.save(
                            temp_path, format="JPEG", optimize=True, quality=quality
                        )
                        file_size = os.path.getsize(temp_path)

                    return temp_path
        except Exception as e:
            logging.error(f"Skipping corrupted/truncated image: {image_path}, {e}")
            return None


def create_model(model_name):
    if model_name == "gpt-4o":
        return GPTModel(model_name)
    elif model_name == "gemini-2.0-flash-exp":
        return GeminiModel(model_name)
    elif model_name == "gemini-1.5-pro-002":
        return GeminiModel(model_name)
    elif model_name == "claude-3-5-sonnet-latest":
        return ClaudeModel(model_name)
    else:
        raise NotImplementedError("Model not supported.")


api_name_to_colloquial = {
    "gpt-4o": "GPT-4o",
    "gemini-2.0-flash-exp": "Gemini 2.0 Flash",
    "gemini-1.5-pro-002": "Gemini 1.5 Pro",
    "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet",
}


class GPTModel(Model):
    def __init__(self, model_name="gpt-4o"):
        self.client = OpenAI()
        self.model_name = model_name
        self.convo = None

    def call_model(self, user_prompt, system_prompt=None, image_paths=None):
        print(
            colored(
                f"\n[Sending request to {api_name_to_colloquial[self.model_name]}...]",
                "yellow",
            )
        )
        if image_paths is not None:
            valid_images = []
            for path in image_paths:
                resized = self.resize_image(path, max_size_mb=20)
                if not resized:
                    logging.warning(f"Skipping image {path}")
                    continue
                valid_images.append(self.encode_image(resized))
            user_prompt = [
                {"type": "text", "text": user_prompt},
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{x}"},
                    },
                    valid_images,
                ),
            ]

        self.convo = OpenAIConversation(
            user_prompt=user_prompt, system_prompt=system_prompt
        )

        MAX_RETRIES = 3
        RETRY_DELAY = 5  # seconds

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.convo.messages,
                )
                response_string = response.choices[0].message.content
                print(colored("\n[GPT Response]", "green"))
                print(colored("=" * 80, "green"))
                print(response_string)
                print(colored("=" * 80, "green"))
                return response_string
            except (APIConnectionError, RateLimitError) as e:
                if attempt == MAX_RETRIES - 1:  # Last attempt
                    raise  # Re-raise the exception if all retries failed
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                continue

    def count_tokens(
        self, system_prompt, user_prompt, assistant_response, image_paths=None
    ):
        encoder = tiktoken.encoding_for_model(self.model_name)

        image_token_count = 0
        if image_paths is not None:
            image_tiles = []
            for image_path in image_paths:
                width, height = self.get_image_dimensions(image_path)
                num_w_tiles = width // 512
                num_w_tiles = num_w_tiles + 1 if width % 512 != 0 else num_w_tiles
                num_h_tiles = height // 512
                num_h_tiles = num_h_tiles + 1 if height % 512 != 0 else num_h_tiles
                num_tiles = num_w_tiles * num_h_tiles
                if num_tiles > 4:
                    num_tiles = 4
                image_tiles.append(num_tiles)

            tokens_per_tile = 170

            image_token_count = sum(
                [num_tiles * tokens_per_tile for num_tiles in image_tiles]
            )

        system_tokens = encoder.encode(system_prompt)
        user_tokens = encoder.encode(user_prompt)
        assistant_tokens = encoder.encode(assistant_response)

        system_token_count = len(system_tokens)
        user_token_count = len(user_tokens)
        assistant_token_count = len(assistant_tokens)

        # Define pricing
        if self.model_name == "gpt-4o":
            price_per_million_input_tokens = 2.50
            price_per_million_output_tokens = 10.00
        elif self.model_name == "gpt-4o-mini":
            price_per_million_input_tokens = 0.15
            price_per_million_output_tokens = 0.60
        else:
            raise NotImplementedError("Pricing not defined for this model.")

        total_input_tokens = system_token_count + user_token_count + image_token_count
        input_cost = (total_input_tokens / 1_000_000) * price_per_million_input_tokens
        output_cost = (
            assistant_token_count / 1_000_000
        ) * price_per_million_output_tokens

        total_cost = input_cost + output_cost

        output_dict = {
            "system_tokens": system_token_count,
            "user_tokens": user_token_count,
            "image_tokens": image_token_count,
            "total_input_tokens": total_input_tokens,
            "input_cost": input_cost,
            "output_tokens": assistant_token_count,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

        return output_dict

    def get_image_dimensions(self, image_path):
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height


class GeminiModel(Model):
    def __init__(self, model_name="gemini-1.5-pro-002"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.warning("[GeminiModel] GOOGLE_API_KEY not set; ensure environment variable is configured.")
        else:
            genai.configure(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = None
        self.convo = None
        self.model = None

    def call_model(self, user_prompt, system_prompt=None, image_paths=None):
        logging.info(f"[GeminiModel] call_model called, model_name={self.model_name}")
        if image_paths:
            logging.info(f"[GeminiModel] Received {len(image_paths)} images to upload.")
        else:
            logging.info("[GeminiModel] No image_paths provided.")

        self.system_prompt = system_prompt
        if self.system_prompt:
            self.model = genai.GenerativeModel(
                model_name=self.model_name, system_instruction=self.system_prompt
            )
        else:
            self.model = genai.GenerativeModel(model_name=self.model_name)

        MAX_RETRIES = 3
        RETRY_DELAY = 2

        if image_paths is not None:
            file_objects = []
            for path in image_paths:
                logging.info(
                    f"[GeminiModel] Attempting to process & upload image: {path}"
                )
                retry_count = 0
                while retry_count < MAX_RETRIES:
                    try:
                        resized = self.resize_image(path, max_size_mb=5)
                        if not resized:
                            logging.warning(
                                f"[GeminiModel] Skipping image {path} because it could not be resized."
                            )
                            break
                        logging.debug(f"[GeminiModel] resized image is at: {resized}")
                        file_obj = genai.upload_file(resized)
                        file_objects.append(file_obj)
                        logging.info(
                            f"[GeminiModel] Successfully uploaded {path} -> object: {file_obj}"
                        )
                        break
                    except Exception as e:
                        retry_count += 1
                        logging.warning(
                            f"[GeminiModel] Failed to upload image {path} attempt {retry_count}/{MAX_RETRIES}: {str(e)}"
                        )
                        if retry_count == MAX_RETRIES:
                            logging.warning(f"[GeminiModel] Giving up on {path}.")
                        else:
                            time.sleep(RETRY_DELAY)
                            continue

            if not file_objects:
                error_msg = "ERROR: No valid images could be uploaded to Gemini"
                logging.error(error_msg)
                return error_msg

            user_prompt = [user_prompt, *file_objects]

        self.convo = GeminiConversation(
            user_prompt=user_prompt, system_prompt=self.system_prompt
        )

        response = None
        for attempt in range(MAX_RETRIES):
            try:
                logging.info(
                    f"[GeminiModel] Generating content (attempt {attempt+1}/{MAX_RETRIES})..."
                )
                response = self.model.generate_content(self.convo.messages)
                break
            except Exception as e:
                logging.error(f"[GeminiModel] Error generating content: {e}")
                if attempt == MAX_RETRIES - 1:
                    error_msg = f"ERROR: Failed to get response from Gemini after {MAX_RETRIES} attempts: {str(e)}"
                    logging.error(error_msg)
                    return error_msg
                time.sleep(RETRY_DELAY * (attempt + 1))

        if not response:
            return "ERROR: No response from Gemini"

        logging.info("[GeminiModel] Successfully got response from Gemini.")
        print(colored("\n[Gemini Response]", "green"))
        print(colored("=" * 80, "green"))
        print(response.text)
        print(colored("=" * 80, "green"))
        return response.text

    def count_tokens(
        self, system_prompt, user_prompt, assistant_response, image_paths=None
    ):
        # Create a GenerativeModel instance with system_instruction for accurate token count
        if system_prompt:
            model = genai.GenerativeModel(
                model_name=self.model_name, system_instruction=system_prompt
            )
        else:
            model = genai.GenerativeModel(model_name=self.model_name)

        # Prepare the input content as in call_model
        if image_paths is not None:
            img_objects = list(map(Image.open, image_paths))
            user_prompt_content = [user_prompt, *img_objects]
        else:
            user_prompt_content = user_prompt

        # Count tokens for the input (system_prompt is included in the model instance)
        input_token_info = model.count_tokens(user_prompt_content)
        total_input_tokens = input_token_info.total_tokens

        # Count tokens for the assistant's response
        assistant_token_info = model.count_tokens(assistant_response)
        assistant_token_count = assistant_token_info.total_tokens

        # Define pricing based on model_name and token counts
        if self.model_name == "gemini-2.0-flash-exp":
            if total_input_tokens <= 128_000:
                price_per_million_input_tokens = 0.075
                price_per_million_output_tokens = 0.30
            else:
                price_per_million_input_tokens = 0.15
                price_per_million_output_tokens = 0.60
        elif self.model_name == "gemini-1.5-pro-002":
            if total_input_tokens <= 128_000:
                price_per_million_input_tokens = 1.25
                price_per_million_output_tokens = 5.00
            else:
                price_per_million_input_tokens = 2.50
                price_per_million_output_tokens = 10.00
        else:
            raise NotImplementedError("Pricing not defined for this model.")

        # Calculate costs
        input_cost = (total_input_tokens / 1_000_000) * price_per_million_input_tokens
        output_cost = (
            assistant_token_count / 1_000_000
        ) * price_per_million_output_tokens

        total_cost = input_cost + output_cost

        output_dict = {
            "total_input_tokens": total_input_tokens,
            "input_cost": input_cost,
            "output_tokens": assistant_token_count,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

        return output_dict


class ClaudeModel(Model):
    def __init__(self, model_name="claude-3-5-sonnet-latest"):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self.convo = None

    def call_model(self, user_prompt, system_prompt=None, image_paths=None):
        print(
            colored(
                f"\n[Sending request to {api_name_to_colloquial[self.model_name]}...]",
                "yellow",
            )
        )
        if image_paths is not None:
            valid_images = []
            for path in image_paths:
                resized = self.resize_image(path, max_size_mb=5)
                if not resized:
                    logging.warning(f"Skipping image {path}")
                    continue
                valid_images.append(self.encode_image(resized))
            user_prompt = [
                {"type": "text", "text": user_prompt},
                *map(
                    lambda x: {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": x,
                        },
                    },
                    valid_images,
                ),
            ]
        self.convo = AnthropicConversation(user_prompt=user_prompt)
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=0.0,
            system=system_prompt,
            messages=self.convo.messages,
        )
        response_string = response.content[0].text
        print(colored("\n[Claude Response]", "green"))
        print(colored("=" * 80, "green"))
        print(response_string)
        print(colored("=" * 80, "green"))
        return response_string

    def count_tokens(
        self, system_prompt, user_prompt, assistant_response, image_paths=None
    ):
        # Use Claude's beta token counting API
        input_content = []

        # Add text content
        if isinstance(user_prompt, str):
            input_content.append({"type": "text", "text": user_prompt})
        else:
            input_content.extend(user_prompt)

        # Add any images
        if image_paths:
            for image_path in image_paths:
                with open(image_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                    input_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64,
                            },
                        }
                    )

        # Get input token count
        input_response = self.client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model=self.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": input_content}],
        )
        total_input_tokens = input_response.input_tokens

        # Get output token count
        output_response = self.client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model=self.model_name,
            messages=[{"role": "assistant", "content": assistant_response}],
        )
        assistant_token_count = output_response.input_tokens

        # Claude 3.5 Sonnet pricing
        price_per_million_input_tokens = 3.00
        price_per_million_output_tokens = 15.00

        input_cost = (total_input_tokens / 1_000_000) * price_per_million_input_tokens
        output_cost = (
            assistant_token_count / 1_000_000
        ) * price_per_million_output_tokens
        total_cost = input_cost + output_cost

        output_dict = {
            "total_input_tokens": total_input_tokens,
            "input_cost": input_cost,
            "output_tokens": assistant_token_count,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

        return output_dict


class OLlamaModel(Model):
    def __init__(self, model_name="llava:34b"):
        self.model_name = model_name
        self.convo = None

    def call_model(self, user_prompt, system_prompt=None, image_paths=None):
        if image_paths is not None:
            encoded_images = list(map(self.encode_image, image_paths))
            user_prompt = {
                "role": "user",
                "content": user_prompt,
                "images": encoded_images,
            }
        else:
            user_prompt = {"role": "user", "content": user_prompt}
        self.convo = OLlamaConversation(
            user_prompt=user_prompt, system_prompt=system_prompt
        )
        json_data = json.dumps(
            {"model": self.model_name, "messages": self.convo.messages, "stream": False}
        )
        result = subprocess.run(
            ["curl", "http://localhost:11434/api/chat", "-d", "@-"],
            input=json_data,
            capture_output=True,
            text=True,
        )
        json_output = result.stdout.strip()
        try:
            data = json.loads(json_output)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None  # Return None or handle the error as needed

        response_string = data["message"]["content"]
        assert response_string is not None, "Make sure OLlama is turned on!"
        return response_string


class RobotVision:
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        goal: str = "",
        open_loop: bool = False,
        state_file: str = "robot_vision_state.json",
    ):
        self.model = create_model(model_name)
        self.system_prompt = system_prompt
        self.goal = goal
        self.open_loop = open_loop
        self.total_cost = 0.0
        self.processed_images: List[dict] = []
        self.state_file = state_file
        self.current_answer = ""
        self.question_count = 0

    def display_images(self, image_paths):
        """Display images directly in the terminal"""
        if not image_paths:
            return

        for i, img_path in enumerate(image_paths, 1):
            print(colored(f"\nImage {i}:", "magenta"))
            print(colored("=" * 80, "magenta"))
            # Create image object and display it
            img = from_file(img_path)
            img.width = min(
                80, os.get_terminal_size().columns
            )  # Adjust to terminal width
            print(img)
            print(colored("=" * 80, "magenta"))

    def print_delimited_section(self, title, content, color="cyan"):
        """Print content with colored delimiters"""
        delimiter = "=" * 80
        print("\n" + colored(delimiter, color))
        print(colored(f"[{title}]", color))
        print(colored(delimiter, color))
        print(content)
        print(colored(delimiter, color) + "\n")

    def print_error(self, model_name: str, error: Exception, attempt: int = None):
        """Print error messages with red formatting"""
        if attempt is not None:
            print(colored(f"\n[{model_name} Error - Attempt {attempt}]", "red"))
        else:
            print(colored(f"\n[{model_name} Error]", "red"))
        print(colored("=" * 80, "red"))
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {str(error)}")
        print(colored("=" * 80, "red"))

    def process_images(self, images_and_annotations: list) -> None:
        """
        Processes all images at once.
        """
        if not self.goal:
            raise ValueError("Goal must be set before processing images")

        # Clear previous state
        self.processed_images = []

        # Add all images and annotations
        for image_path, annotation in images_and_annotations:
            if not os.path.exists(image_path):
                print(colored(f"\n[Error] Image {image_path} does not exist.", "red"))
                continue
            self.processed_images.append(
                {"image_path": image_path, "annotation": annotation}
            )

        # Prepare the prompt and images
        annotations_text = "\n".join(
            [
                f"Image {idx + 1} annotation: {item['annotation']}"
                for idx, item in enumerate(self.processed_images)
                if item["annotation"]
            ]
        )
        combined_prompt = f"{self.goal}\n\nAnnotations:\n{annotations_text}"
        image_paths = [item["image_path"] for item in self.processed_images]

        # Print question information
        self.print_delimited_section(
            f"Question {self.question_count}", f"Goal: {self.goal}", "yellow"
        )

        # Display system prompt
        self.print_delimited_section("System Prompt", self.system_prompt, "green")

        # Display annotations if any
        if annotations_text:
            self.print_delimited_section("Annotations", annotations_text, "blue")

        # Display images
        print(colored("\nDisplaying images...", "magenta"))
        self.display_images(image_paths)

        # Try all available models
        for model_name in api_name_to_colloquial.keys():
            MAX_RETRIES = 3
            RETRY_DELAY = 5  # seconds
            success = False

            for attempt in range(MAX_RETRIES):
                try:
                    model = create_model(model_name)
                    result = model.call_model(
                        user_prompt=combined_prompt,
                        system_prompt=self.system_prompt,
                        image_paths=image_paths,
                    )
                    success = True
                    break
                except (APIConnectionError, RateLimitError) as e:
                    self.print_error(api_name_to_colloquial[model_name], e, attempt + 1)
                    if attempt == MAX_RETRIES - 1:
                        print(
                            colored(
                                f"\n[{api_name_to_colloquial[model_name]} Failed] Maximum retries reached.",
                                "red",
                            )
                        )
                    else:
                        wait_time = RETRY_DELAY * (attempt + 1)
                        print(
                            colored(
                                f"Waiting {wait_time} seconds before retry...", "yellow"
                            )
                        )
                        time.sleep(wait_time)
                except Exception as e:
                    self.print_error(api_name_to_colloquial[model_name], e)
                    break

            if success:
                self.current_answer = result

        # Save the current state after processing
        self.save_state()

    def get_analysis(self, all_at_once=False) -> str:
        """
        Processes images and returns the LLM's analysis.
        """
        if not self.goal:
            raise ValueError("Goal must be set before processing images")

        self.question_count += 1

        # Prepare annotations corresponding to images
        annotations_text = "\n".join(
            [
                f"Image {idx + 1} annotation: {item['annotation']}"
                for idx, item in enumerate(self.processed_images)
                if item["annotation"]
            ]
        )

        # Combine the goal with annotations
        combined_prompt = f"{self.goal}\n\nAnnotations:\n{annotations_text}"

        image_paths = [item["image_path"] for item in self.processed_images]

        # Print question information with colored delimiters
        self.print_delimited_section(
            f"Question {self.question_count}", f"Goal: {self.goal}", "yellow"
        )

        # Display system prompt
        self.print_delimited_section("System Prompt", self.system_prompt, "green")

        # Display annotations if any
        if annotations_text:
            self.print_delimited_section("Annotations", annotations_text, "blue")

        # Display images
        print(colored("\nDisplaying images...", "magenta"))
        self.display_images(image_paths)

        if all_at_once:
            # Call the model with all images and the combined prompt
            result = self.model.call_model(
                user_prompt=combined_prompt,
                system_prompt=self.system_prompt,
                image_paths=image_paths,
            )
        else:
            # Call the model with the last image and the combined prompt
            result = self.model.call_model(
                user_prompt=combined_prompt,
                system_prompt=self.system_prompt,
                image_paths=[image_paths[-1]],
            )

        # Display model response
        self.print_delimited_section(
            f"Model Response ({api_name_to_colloquial[self.model.model_name]})",
            result,
            "red",
        )

        # Save the current answer in the state
        self.current_answer = result

        # Save the current state after processing
        self.save_state()

        return result

    def get_output(self) -> str:
        """
        Returns the current answer from the LLM.
        """
        return self.current_answer

    def save_state(self):
        """
        Saves the current state (processed_images, total_cost, system_prompt, and current_answer) to a JSON file.
        """
        state = {
            "goal": self.goal,
            "system_prompt": self.system_prompt,
            "total_cost": self.total_cost,
            "processed_images": self.processed_images,
            "current_answer": self.current_answer,
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)
            print(f"State saved to {self.state_file}.")
        except Exception as e:
            print(f"Failed to save state to {self.state_file}: {e}")

    def set_goal(self, goal: str):
        self.goal = goal

    def process_new_image(self, image_path: str, annotation: str = "") -> None:
        if not self.goal:
            raise ValueError("Goal must be set before processing images")

        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist.")
            return

        # Add the new image and annotation to the processed_images list
        self.processed_images.append(
            {"image_path": image_path, "annotation": annotation}
        )

        # Get the LLM's analysis
        self.get_analysis()
