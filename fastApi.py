import numpy as np
import cv2
from PaddleOCR import answer
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List

app = FastAPI()
def test(img):
    return answer.question_answer(img)
@app.post("/chat/")
async def chat(input_img: UploadFile = File(...)):
    # Save the uploaded image
    with open("input.png", "wb") as buffer:
        buffer.write(await input_img.read())

    # Read the saved image
    img = cv2.imread("input.png")

    # Process the image and get the response
    response = test(img)

    return JSONResponse(content={"response": response})


# To run the server, you would use the command: uvicorn fastApi:app --reload
