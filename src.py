import os
import numpy as np
import openai
from openai import OpenAI
from typing import List, Tuple, Dict, Any, Union

from PaddleOCR.predict_det import run_text_detector
from PaddleOCR.cut import sort_bbox, crop_image_with_boxes
from PaddleOCR.predict_rec import run_text_rec


def chat_with_openai(client: openai.OpenAI, model_name: str, system_prompt: str, prompt: str, history: List[Dict[str, str]] = []) -> openai.Completion:
    """
    Sends the prompt to OpenAI API using the chat interface and retrieves the model's response.

    Parameters:
    - client (openai.OpenAI): The OpenAI API client instance.
    - model_name (str): The name of the model to use for the chat.
    - system_prompt (str): The initial system prompt that sets the context for the chat.
    - prompt (str): The user's input prompt to the chat model.
    - history (List[Dict[str, str]]): A list of previous message exchanges, if any.

    Returns:
    - openai.Completion: The chat model's response.
    """
    message = {'role': 'user', 'content': prompt}
    system_message = {"role": "system", "content": system_prompt}
    messages = [system_message] + history + [message]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    return chat_completion


class AIStudyController:
    @staticmethod
    async def ocr_detect(img: np.ndarray, use_gpu: bool = False) -> List[Tuple[float]]:
        """
        Detects text within an image using PaddleOCR's detection model.

        Parameters:
        - img (np.ndarray): The image array.
        - use_gpu (bool, optional): Flag to use GPU acceleration.

        Returns:
        - List[Tuple[float]]: Sorted bounding boxes for detected text areas.
        """
        det_result = run_text_detector(img=img, use_gpu=use_gpu, det_model_dir="PaddleOCR/en_PP-OCRv3_det_infer/")
        bounding_boxes = eval(det_result[0])
        sorted_boxes = sort_bbox(bounding_boxes)
        return sorted_boxes

    @staticmethod
    async def ocr_recognize(img: np.ndarray, bounding_boxes: List[Tuple[float]], use_gpu: bool = False) -> List[str]:
        """
        Recognizes text from given bounding boxes in an image.

        Parameters:
        - img (np.ndarray): The image array.
        - bounding_boxes (List[Tuple[float]]): Bounding boxes for text areas.
        - use_gpu (bool, optional): Flag to use GPU acceleration.

        Returns:
        - List[str]: Predicted text for each bounding box.
        """
        cropped_images = crop_image_with_boxes(img, bounding_boxes)
        predicted_text = run_text_rec(cropped_images, use_gpu, "PaddleOCR/ppocr/utils/en_dict.txt", "PaddleOCR/en_PP-OCRv3_rec_infer/")
        return predicted_text

    @staticmethod
    async def parse_question(predicted_text: List[str]) -> str:
        """
        Concatenates recognized text items into a single question string.

        Parameters:
        - predicted_text (List[str]): The recognized text items.

        Returns:
        - str: The concatenated question string.
        """
        question = " ".join([item[0] for item in predicted_text])
        return question

    @staticmethod
    async def retrieval_llm(question: str, model_name: str, system_prompt: str, openai_api_key: str) -> str:
        """
        Retrieves an answer for a given question from an LLM (Large Language Model) using the OpenAI API.

        Parameters:
        - question (str): The question to ask the LLM.
        - model_name (str): The name of the model to use for the retrieval.
        - system_prompt (str): The initial system prompt for the chat session.
        - openai_api_key (str): The API key for authenticating with the OpenAI API.

        Returns:
        - str: The retrieved answer.
        """
        client = openai.OpenAI(api_key=openai_api_key)
        prompt = f"QUESTION: {question}"
        result = chat_with_openai(client, model_name, system_prompt, prompt, [])
        answer = result.choices[0].message.content
        return answer

    @staticmethod
    async def run(img: np.ndarray, configs: Dict[str, Any] = {}) -> Union[str, None]:
        """
        Executes the full pipeline from OCR detection and recognition to question parsing and LLM retrieval.

        Parameters:
        - img (np.ndarray): The image to process.
        - configs (Dict[str, Any], optional): Configuration options including GPU usage, model name, system prompt, and OpenAI API key.

        Returns:
        - Union[str, None]: The answer retrieved from the LLM, or None if an error occurs.
        """
        use_gpu = configs.get('use_gpu', False)
        model_name = configs.get('model_name', '')
        system_prompt = configs.get('system_prompt', '')
        openai_api_key = configs.get('openai_api_key', '')

        bounding_boxes = await AIStudyController.ocr_detect(img, use_gpu)
        predicted_text = await AIStudyController.ocr_recognize(img, bounding_boxes, use_gpu)
        question = await AIStudyController.parse_question(predicted_text)
        answer = await AIStudyController.retrieval_llm(question, model_name, system_prompt, openai_api_key)

        return answer
