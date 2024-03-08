import matplotlib.pyplot as plt
from PaddleOCR.predict_det import *
from PaddleOCR.cut import *
from PaddleOCR.predict_rec import *
from PaddleOCR.prompt import *
def question_answer(img):
    det_result=run_text_detector(
            img=img,
            use_gpu=False,
            det_model_dir="PaddleOCR/en_PP-OCRv3_det_infer/",
        )
    bounding_box_str=det_result[0]
    bounding_boxes = eval(bounding_box_str)
    bounding_boxes=sort_bbox(bounding_boxes)
    cut_images=crop_image_with_boxes(img, bounding_boxes)
    text_predict=run_text_rec(cut_images,False,"PaddleOCR/ppocr/utils/en_dict.txt","PaddleOCR/en_PP-OCRv3_rec_infer/")
    question = " ".join([item[0] for item in text_predict])
    # print(question)
    return answer(question)
# print(question_answer(cv2.imread("3.png")))