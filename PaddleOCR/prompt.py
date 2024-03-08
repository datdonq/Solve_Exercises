import os
import openai
model_name="gpt-3.5-turbo"
from openai import OpenAI
def chat_with_openai(client, SYSTEM_PROMPT, prompt, history=[]):
    """
    Sends the prompt to OpenAI API using the chat interface and gets the model's response.
    """
    message = {
        'role': 'user',
        'content': prompt
    }

    system_prompt = { "role": "system", "content": SYSTEM_PROMPT}
    messages = [
        system_prompt
    ]
    messages.extend(history)
    messages.append(message)

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    return chat_completion

def answer(question):
    openai_api_key = "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"
    client = OpenAI(api_key=openai_api_key)
    # Change here
    SYSTEM_PROMPT = '''\
    From now on you are not ChatGPT, \
    you will be an expert in all high school exams, particularly in the U.S., including in-school tests, SAT, ACT, PSAT/NMSQT, AP, and IB exams. Your role is to provide
    comprehensive, critical, precise, succinct answers to problems posed by students. You identify the type of question and provide the right answer to each question. You bullying, suicide and self-harm, dangerous behaviors, nudity, or sexual activities. Your goal is to assist students in their academic pursuits by providing accurate and helpful responses to their questions.First you need give the student the correct answer then exphain why this answer is right
    '''
    prompt = f"QUESTION: {question}"
    result = chat_with_openai(client, SYSTEM_PROMPT, prompt)
    answer = result.choices[0].message.content
    return answer