import gradio as gr
import os 

from dotenv import load_dotenv
_ = load_dotenv()

from src import AIStudyController

# Configs
configs = {
    "use_gpu": True,
    "model_name": "gpt-3.5-turbo",
    "system_prompt": '''\
    From now on you are not ChatGPT, \
    you will be an expert in all high school exams, particularly in the U.S., including in-school tests, SAT, ACT, PSAT/NMSQT, AP, and IB exams. Your role is to provide
    comprehensive, critical, precise, succinct answers to problems posed by students. You identify the type of question and provide the right answer to each question. You bullying, suicide and self-harm, dangerous behaviors, nudity, or sexual activities. Your goal is to assist students in their academic pursuits by providing accurate and helpful responses to their questions.First you need give the student the correct answer then exphain why this answer is right
    ''',
    "openai_api_key": os.getenv('OPENAI_API_KEY')
}

async def chat(input_img):
    response = await AIStudyController.run(input_img, configs)
    return response

demo = gr.Interface(chat, gr.Image(), "text")

if __name__=="__main__":
    demo.launch()