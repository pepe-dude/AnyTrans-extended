import numpy as np

from util import *

import os
import re
import random

from modelscope.pipelines import pipeline
from paddleocr import PaddleOCR
import argparse
import cv2
from difflib import SequenceMatcher
from PIL import Image
from openai import OpenAI, APIStatusError

from typing import List


# compatibility fix for numpy
if not hasattr(np, "int"):
    np.int = int


MODEL = "deepseek-v3.2"
BASE_URL = "https://llm.ai.e-infra.cz/v1/"


promptTemplates = {
    "cs2en": 'Translate the following phrase from Czech into English.\Czech: <box1>Bílý tygr</box1>\nEnglish: <box1>white tiger</box1>\n\nTranslate the following sentence from Czech into English.\Czech: <box1>Všechno nejlepší</box1><box2>k narozeninám</box2>\nEnglish: <box1>Happy</box1><box2>Birthday</box2>\n\nTranslate the following sentence from Czech into English.\Czech: <box1>Šťastný</box1><box2>nový rok</box2>\nEnglish: <box1>Happy</box1><box2>New Year</box2>\n\nTranslate the following sentence from Czech into English.\Czech: <box1>@Doprava Peking</box1><box2>Shentuo č. 2</box2><box3>Pekingská skupina pro údržbu</box3>\nEnglish: <box1>@Beijing Transportation</box1><box2>Shen Tuo No. 2</box2><box3>Beijng Maintenance Group</box3>\n\nTranslate the following sentence from Czech into English.\Czech: <box1>Prosím, važte si</box1><box2>svěží zelené trávy</box2>\nEnglish: <box1>Grass is green and fresh</box1><box2>please cherish it</box2>\n\nTranslate the following sentence from Czech into English.\Czech: <box1>Brána</box1><box2>Dubové listy</box2><box3>Pokladna</box3><box4>Vstup se psy zakázán</box4><box5>Zákaz kouření</box5><box6>Vyšší než já</box6><box7>Koupit lístek</box7>\nEnglish: <box1>Gate</box1><box2>oak leaves</box2><box3>Ticket Counter</box3><box4>No Dogs Allowed</box4><box5>No Smoking</box5><box6>Higher than me</box6><box7>Buy tickets</box7>\n\nTranslate the following sentence from Czech into English, keep the length similar and no more than 20 letters.',
    "en2cs": 'Translate the following phrase from English into Czech.\English: <box1>white tiger</box1>\nCzech: <box1>Bílý tygr</box1>\n\nTranslate the following sentence from English into Czech.\English: <box1>Happy</box1><box2>Birthday</box2>\nCzech: <box1>Všechno nejlepší</box1><box2>k narozeninám</box2>\n\nTranslate the following sentence from English into Czech.\English: <box1>Happy</box1><box2>New Year</box2>\nCzech: <box1>Šťastný</box1><box2>nový rok</box2>\n\nTranslate the following sentence from English into Czech.\English: <box1>@Beijing Transportation</box1><box2>Shen Tuo No. 2</box2><box3>Beijng Maintenance Group</box3>\nCzech: <box1>@Doprava Peking</box1><box2>Shentuo č. 2</box2><box3>Pekingská skupina pro údržbu</box3>\n\nTranslate the following sentence from English into Czech.\English: <box1>Grass is green and fresh</box1><box2>please cherish it</box2>\nCzech: <box1>Prosím, važte si</box1><box2>svěží zelené trávy</box2>\n\nTranslate the following sentence from English into Czech.\English: <box1>Gate</box1><box2>oak leaves</box2><box3>Ticket Counter</box3><box4>No Dogs Allowed</box4><box5>No Smoking</box5><box6>Higher than me</box6><box7>Buy tickets</box7>\nCzech: <box1>Brána</box1><box2>Dubové listy</box2><box3>Pokladna</box3><box4>Vstup se psy zakázán</box4><box5>Zákaz kouření</box5><box6>Vyšší než já</box6><box7>Koupit lístek</box7>\n\nTranslate the following sentence from English into Czech, keep the length similar and no more than 20 letters.',
    
    "cs2uk": 'Translate the following phrase from Czech into Ukranian.\Czech: <box1>Bílý tygr</box1>\nUkranian: <box1>Білий тигр</box1>\n\nTranslate the following sentence from Czech into Ukranian.\Czech: <box1>Všechno nejlepší</box1><box2>k narozeninám</box2>\nUkranian: <box1>З днем</box1><box2>народження</box2>\n\nTranslate the following sentence from Czech into Ukranian.\Czech: <box1>Šťastný</box1><box2>nový rok</box2>\nUkranian: <box1>З</box1><box2>Новим роком</box2>\n\nTranslate the following sentence from Czech into Ukranian.\Czech: <box1>@Doprava Peking</box1><box2>Shentuo č. 2</box2><box3>Pekingská skupina pro údržbu</box3>\nUkranian: <box1>@Транспорт Пекін</box1><box2>Шеньтуо № 2</box2><box3>Пекінська група технічного обслуговування</box3>\n\nTranslate the following sentence from Czech into Ukranian.\Czech: <box1>Prosím, važte si</box1><box2>svěží zelené trávy</box2>\nUkranian: <box1>Будь ласка</box1><box2>оцініть пишну зелену траву</box2>\n\nTranslate the following sentence from Czech into Ukranian.\Czech: <box1>Brána</box1><box2>Dubové listy</box2><box3>Pokladna</box3><box4>Vstup se psy zakázán</box4><box5>Zákaz kouření</box5><box6>Vyšší než já</box6><box7>Koupit lístek</box7>\nUkranian: <box1>Ворота</box1><box2>дубове листя</box2><box3>Каса</box3><box4>Вхід з собаками заборонено</box4><box5>Куріння заборонено</box5><box6>Вищий за мене</box6><box7>Купити квиток</box7>\n\nTranslate the following sentence from Czech into Ukranian, keep the length similar and no more than 20 letters.',
    "uk2cs": 'Translate the following phrase from Ukranian into Czech.\\Ukranian: <box1>Білий тигр</box1>\nCzech: <box1>Bílý tygr</box1>\n\nTranslate the following sentence from Ukranian into Czech.\\Ukranian: <box1>З днем</box1><box2>народження</box2>\nCzech: <box1>Všechno nejlepší</box1><box2>k narozeninám</box2>\n\nTranslate the following sentence from Ukranian into Czech.\\Ukranian: <box1>З</box1><box2>Новим роком</box2>\nCzech: <box1>Šťastný</box1><box2>nový rok</box2>\n\nTranslate the following sentence from Ukranian into Czech.\\Ukranian: <box1>@Транспорт Пекін</box1><box2>Шеньтуо № 2</box2><box3>Пекінська група технічного обслуговування</box3>\nCzech: <box1>@Doprava Peking</box1><box2>Shentuo č. 2</box2><box3>Pekingská skupina pro údržbu</box3>\n\nTranslate the following sentence from Ukranian into Czech.\\Ukranian: <box1>Будь ласка</box1><box2>оцініть пишну зелену траву</box2>\nCzech: <box1>Prosím, važte si</box1><box2>svěží zelené trávy</box2>\n\nTranslate the following sentence from Ukranian into Czech.\\Ukranian: <box1>Ворота</box1><box2>дубове листя</box2><box3>Каса</box3><box4>Вхід з собаками заборонено</box4><box5>Куріння заборонено</box5><box6>Вищий за мене</box6><box7>Купити квиток</box7>\nCzech: <box1>Brána</box1><box2>Dubové listy</box2><box3>Pokladna</box3><box4>Vstup se psy zakázán</box4><box5>Zákaz kouření</box5><box6>Vyšší než já</box6><box7>Koupit lístek</box7>\n\nTranslate the following sentence from Ukranian into Czech, keep the length similar and no more than 20 letters.',

    "cs2german": 'Translate the following phrase from Czech into German.\Czech: <box1>Bílý tygr</box1>\nGerman: <box1>Weißer Tiger</box1>\n\nTranslate the following sentence from Czech into German.\Czech: <box1>Všechno nejlepší</box1><box2>k narozeninám</box2>\nGerman: <box1>Alles Gute</box1><box2>zum Geburtstag</box2>\n\nTranslate the following sentence from Czech into German.\Czech: <box1>Šťastný</box1><box2>nový rok</box2>\nGerman: <box1>Frohes</box1><box2>Neues Jahr</box2>\n\nTranslate the following sentence from Czech into German.\Czech: <box1>@Doprava Peking</box1><box2>Shentuo č. 2</box2><box3>Pekingská skupina pro údržbu</box3>\nGerman: <box1>@Pekinger Verkehr</box1><box2>Shentuo Nr. 2</box2><box3>Wartungsgruppe Peking</box3>\n\nTranslate the following sentence from Czech into German.\Czech: <box1>Prosím, važte si</box1><box2>svěží zelené trávy</box2>\nGerman: <box1>Bitte bewundern</box1><box2>Sie das saftige Grün des Grases</box2>\n\nTranslate the following sentence from Czech into German.\Czech: <box1>Brána</box1><box2>Dubové listy</box2><box3>Pokladna</box3><box4>Vstup se psy zakázán</box4><box5>Zákaz kouření</box5><box6>Vyšší než já</box6><box7>Koupit lístek</box7>\nGerman: <box1>Eichenlaub</box1><box2>Tor</box2><box3>Kasse</box3><box4>Hunde nicht erlaubt</box4><box5>Rauchen verboten</box5><box6>Größer als ich</box6><box7>Kaufen Sie ein Ticket</box7>\n\nTranslate the following sentence from Czech into German, keep the length similar and no more than 20 letters.',
    "german2cs": 'Translate the following phrase from German into Czech.\German: <box1>Weißer Tiger</box1>\nCzech: <box1>Bílý tygr</box1>\n\nTranslate the following sentence from German into Czech.\German: <box1>Alles Gute</box1><box2>zum Geburtstag</box2>\nCzech: <box1>Všechno nejlepší</box1><box2>k narozeninám</box2>\n\nTranslate the following sentence from German into Czech.\German: <box1>Frohes</box1><box2>Neues Jahr</box2>\nCzech: <box1>Šťastný</box1><box2>nový rok</box2>\n\nTranslate the following sentence from German into Czech.\German: <box1>@Pekinger Verkehr</box1><box2>Shentuo Nr. 2</box2><box3>Wartungsgruppe Peking</box3>\nCzech: <box1>@Doprava Peking</box1><box2>Shentuo č. 2</box2><box3>Pekingská skupina pro údržbu</box3>\n\nTranslate the following sentence from German into Czech.\German: <box1>Bitte bewundern</box1><box2>Sie das saftige Grün des Grases</box2>\nCzech: <box1>Prosím, važte si</box1><box2>svěží zelené trávy</box2>\n\nTranslate the following sentence from German into Czech.\German: <box1>Eichenlaub</box1><box2>Tor</box2><box3>Kasse</box3><box4>Hunde nicht erlaubt</box4><box5>Rauchen verboten</box5><box6>Größer als ich</box6><box7>Kaufen Sie ein Ticket</box7>\nCzech: <box1>Brána</box1><box2>Dubové listy</box2><box3>Pokladna</box3><box4>Vstup se psy zakázán</box4><box5>Zákaz kouření</box5><box6>Vyšší než já</box6><box7>Koupit lístek</box7>\n\nTranslate the following sentence from German into Czech, keep the length similar and no more than 20 letters.',
}


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=BASE_URL
)

parser = argparse.ArgumentParser()
parser.add_argument("--sourceLang", default="cs", choices=["cs", "en", "uk", "german"])
parser.add_argument("--targetLang", default="en", choices=["cs", "en", "uk", "german"])
parser.add_argument("--rendering", default="generative", choices=["generative", "deterministic"])
parser.add_argument("--log", type=bool, default=False)


pipe_params = {
    "show_debug": True,
    "image_count": 1,
    "ddim_steps": 20,
}


def call_with_prompt_llm_all_boxbybox(inputText, trans_mode):
    outputText='<box1>'+inputText+'</box1>'
    promptTemplate = promptTemplates[trans_mode] + outputText
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": promptTemplate,
                },
            ]
        )
        
        resultTxt = response.choices[0].message.content
        boxPattern = re.compile(r'<box\d+>(.*?)</box\d+>')

        matches = boxPattern.findall(resultTxt)
        if len(matches) == 1:
            resultTxt = matches[0]
            return resultTxt
        else:
            return resultTxt
        
    except APIStatusError:
        resultTxt = 'failed'
        return resultTxt 
    
    except Exception:
        resultTxt = 'failed'
        return resultTxt


def call_with_prompt_llm_all(inputText: List[str], trans_mode):
    # craft the LLM prompt
    outputText = ''.join([f'<box{i+1}>{text}</box{i+1}>' for i, text in enumerate(inputText)])
    promptTemplate = promptTemplates[trans_mode] + outputText
    
    try:
        # make the API call
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": promptTemplate,
                },
            ]
        )
        
        resultTxt = response.choices[0].message.content
        boxPattern = re.compile(r'<box\d+>(.*?)</box\d+>')

        # check for correct format
        matches = boxPattern.findall(resultTxt)
        if len(inputText) == len(matches):
            resultTxt = matches

            return True, resultTxt
        
        elif len(inputText) == 1:
            return True, [resultTxt]
        
        else:
            return False, resultTxt
        
    except APIStatusError as e:
        resultTxt = 'failed'
        return resultTxt 
    
    except Exception as e:
        resultTxt = 'failed'
        return resultTxt
    

def translaor_using_piple(input_txt, trans_mode):
    # translate each text box individually and return the result
    responses_all = []
    for tmp_txt in input_txt:
        tmp_response = call_with_prompt_llm_all_boxbybox(tmp_txt, trans_mode)
        responses_all.append(tmp_response)

    return True,responses_all 


def render_text_generative(pipe, input_data, response, return_box, evaluate_ocr, idx, max_attempts=5):
    '''
        Use the AnyText pipeline to edit the image by inserting the translated text in place of the original
        by making several attempts - each time checking if the text in the resulting image matches the one 
        returned in the translation step using OCR on that resulting image.
    '''
    
    image = None
    detected_text = ""
    response = response.strip()
    for attempt in range(max_attempts):
        if attempt > 0:
            input_data["seed"] = random.randint(1, 10_000_000)

        results, rtn_code, _, _ = pipe(input_data, mode="text-editing", **pipe_params)

        if rtn_code < 0:
            continue

        image = cv2.cvtColor(results[0], cv2.COLOR_RGB2BGR)

        crop_image_path = image_crop(image, return_box, None, idx)
        crop_result = evaluate_ocr.ocr(crop_image_path)

        if crop_result[0] == None:
            detected_text = ""

        else:
            detected_text = " ".join([line[1][0] for line in crop_result])
        
        # Compare the resulting text from the new image with the translation and if most of it matches return the result
        if SequenceMatcher(None, detected_text, response).ratio() > 0.85:
            return image, detected_text
    
    return image, detected_text


def render_text_deterministic(image, box, text):
    '''Deterministically render the translated text into the image by using PIL to draw it.'''
    font_path = "DejaVuSans.ttf"

    box = np.array(box)

    angle = get_box_angle(box)
    centre = box[:,0].mean(), box[:,1].mean()
    box_width, box_height = get_box_dimentions(box)
    font_size = find_optimal_font_size(text, box_width, box_height, font_path)

    text_img = create_text_image(text, font_path, font_size)

    rotated_text = text_img.rotate(-angle, expand=True, resample=Image.BICUBIC)

    result = paste_rotated_text(image, rotated_text, centre)

    return result


def PPOCR_pipline(img_path, ocr, evaluate_ocr, sourceLang, targetLang, rendering, log):
    trans_mode = f'{sourceLang}2{targetLang}'

    # Text Detection & Recognition
    pil_image = cv2.imread(img_path)
    result = ocr.ocr(img_path)
    
    # Detected text boxes
    dt_boxes = [line[0] for line in result]

    # Adjust the image boxes
    pil_image, new_dt_boxes = resize_image_boxes(pil_image,dt_boxes, max_length=768)    
    resize_image_path = img_path[:-4] + '_resize.jpg'

    cv2.imwrite(resize_image_path, pil_image)
    img_path = resize_image_path

    for i,tmp_box in enumerate(new_dt_boxes):
        new_dt_boxes[i] = enlarge_box_bigger(tmp_box)

    # Detected text
    all_text_ocr = [line[1][0] for line in result]

    if all_text_ocr == []:
        return None

    # Translation of the detected text via API calls to an LLM
    judge, translate_responses = call_with_prompt_llm_all(all_text_ocr, trans_mode)
    
    # If translation of all image boxes at once fails, attempt translation of the boxes one by one
    if judge == False:
        judge,translate_responses = translaor_using_piple(all_text_ocr, trans_mode)

    # If both translation methods fail - write a log and return
    if judge == False:
        if log:
            translation_log = img_path[:-4] + "_wrongtranslation_log.txt"
            log_file = open(translation_log, "w")
            log_file.write(str(all_text_ocr) + "\t")
            log_file.write(str(translate_responses) + "\t")        
        
        return None

    image = np.array(pil_image)
    image = image.clip(0, 255) 
    ori_image_path = img_path

    if log:
        translation_log = img_path[:-4] + "translation_log.txt"
        log_file = open(translation_log, "w", encoding="utf-8")
        evaluate_log = img_path[:-4] + "evaluate_log.txt"
        evaluate_log_file = open(evaluate_log, "w", encoding="utf-8")

    # Proceed to editing of each text box
    for idx in range(len(new_dt_boxes)):
        boxes = new_dt_boxes[idx]
        char_count_ori = len(all_text_ocr[idx])
        tmp_trans = all_text_ocr[idx]
        untranslate = False

        # Check if the translated text contains non translatable text (such as codes number or symbols)
        letterPattern = re.compile(r'[^\W\d_]', re.UNICODE)
        if not re.search(letterPattern, tmp_trans):
            untranslate = True

        try :
            char_count_trans = len(translate_responses[idx])
        except:
            return None
        
        # Rescale the text box in case the length of the original and translated text varies too much
        resized_mask, return_box = resize_mask_returnbox(ori_image_path,boxes,char_count_ori,char_count_trans)
        resized_masked_image = cv2.bitwise_and(image, image, mask=resized_mask)

        # masked_image_path = img_path[:-4] + '_masked_'+str(idx) + '.png'
        resized_masked_image_path = img_path[:-4]+'_resized_masked.png'

        cv2.imwrite(resized_masked_image_path, resized_masked_image)

        # If the orignal text has been marked for inpainting and it is translatable (does not contain only symbols, numbers or codes)
        if untranslate == False:
            inpaint_image(ori_image_path, boxes)

        txts = all_text_ocr[idx]

        if log:
            log_file.write(txts + "\t")

        response = translate_responses[idx]
        
        # Check if the translated result is the same as the original text in which case there is no need for text replacement
        edited_image_path = img_path[:-4] + '_edited.png'
        if response == txts:
            cv2.imwrite(edited_image_path, image)
            edited_image = image

            if log:
                evaluate_log_file.write(f"{txts}\t{response}\t{response}\n")

        elif untranslate == True: # In case the text is untranslatable (text box contains only numbers or codes)
            cv2.imwrite(edited_image_path, image)
            edited_image = image

            if log:
                evaluate_log_file.write(f"{txts}\t{tmp_trans}\t{tmp_trans}\n")

        else: # Text needs to be edited
            if rendering == "generative":
                tmp_prompt = f'The image contains ONLY the text "{response}" written clearly and correctly'
                input_data = {
                    "prompt": tmp_prompt,
                    "seed": 94081527,
                    "draw_pos":resized_masked_image_path,
                    "ori_image":ori_image_path,
                }
                
                edited_image, detected_text = render_text_generative(pipe, input_data, response, return_box, evaluate_ocr, idx)                

                if log:
                    evaluate_log_file.write(f"{txts}\t{response}\t{detected_text}\n")

            elif rendering == "deterministic":
                source_image = cv2.imread(ori_image_path)
                edited_image = render_text_deterministic(source_image, return_box, response)

            cv2.imwrite(edited_image_path, edited_image)

        ori_image_path = edited_image_path
        image = edited_image.clip(0, 255)


def main(args: argparse.Namespace):
    folder_path = '/home/pepe/Documents/University year 4/Competing in machine translation/AnyTrans-extended/cs_2_all/cs2en'
    image_path_list = []
    ocr = PaddleOCR(lang = args.sourceLang)
    evaluate_ocr = PaddleOCR(lang = args.targetLang)

    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):          
            image_path = os.path.join(folder_path, filename)
            image_path_list.append(image_path)

    for i in range(1):
        path = os.path.join(folder_path,f'cs_{i+1}.png')
        if path in image_path_list:
            print(f"Processed image: {path}")

            PPOCR_pipline(path, ocr, evaluate_ocr, args.sourceLang, args.targetLang, args.rendering, args.log)


if __name__ == '__main__':
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    
    if main_args.rendering == "generative":
        pipe = pipeline('my-anytext-task', model='damo/cv_anytext_text_generation_editing', model_revision='v1.1.2')

    main(main_args)