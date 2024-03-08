import numpy as np
import gradio as gr
import cv2
from PaddleOCR import answer
def chat(input_img):
    return answer.question_answer(input_img)
# inputs = [gr.Image(), gr.Textbox(label="Language")]
demo = gr.Interface(chat, gr.Image(), "text")
demo.launch()
# print(answer.question_answer(cv2.imread("2.png")))