from typing import Dict
import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR

from dotenv import load_dotenv
_ = load_dotenv()

from src import AIStudyController

# Init app
app = FastAPI()

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


# Endpoints
@app.post("/chat/", response_model=Dict[str, str])
async def chat(input_img: UploadFile = File(...)) -> JSONResponse:

    contents = await input_img.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        response = await AIStudyController.run(img, configs)
    except ValueError as e:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing the image: {e}")

    return JSONResponse(content={"response": response})

