import numpy as np

from util import resize_image,resize_mask,enlarge_box_bigger,image_crop,resize_mask_returnbox_suokuan,resize_image_boxes
from util import save_images

import os
import re
import sys
import random
import time

from modelscope.pipelines import pipeline
from paddleocr import PaddleOCR, draw_ocr
from ppocr_pipline_alibabacatu import Alicatu
from cv_inpainting import InpaintImage
from http import HTTPStatus
import argparse
import cv2
import dashscope
import torch
from openai import OpenAI, APIStatusError

from typing import List

# compatibility fix for numpy
if not hasattr(np, "int"):
    np.int = int

MODEL = "deepseek-v3.2"
BASE_URL = "https://llm.ai.e-infra.cz/v1/"

# TODO: setup a dictionary with prompt templates for each language pair
# TODO: Leave the end open so that the translated text can be inserted
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

params = {
    "show_debug": True,
    "image_count": 1,
    "ddim_steps": 20,
}

# pipe = pipeline('my-anytext-task', model='damo/cv_anytext_text_generation_editing', model_revision='v1.1.2')
# pipe = pipeline(
#     'my-anytext-task',
#     model='damo/cv_anytext_text_generation_editing',
#     device='cpu',   # force CPU
# )


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
        if len(matches)==1:
            resultTxt=matches[0]
            return resultTxt
        else:
            return resultTxt
        
    except APIStatusError:
        resultTxt='failed'
        return resultTxt 
    
    except Exception:
        resultTxt='failed'
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
        if len(inputText)==len(matches):
            resultTxt = matches
            return True,resultTxt
        elif len(inputText)==1:
            return True,[resultTxt]
        else:
            return False, resultTxt
        
    except APIStatusError as e:
        resultTxt='failed'
        return resultTxt 
    
    except Exception as e:
        resultTxt='failed'
        return resultTxt
    

def translaor_using_piple(input_txt, sourceLang, targetLang):
    # translate each text box individually and return the result
    responses_all=[]
    for tmp_txt in input_txt:
        tmp_response=call_with_prompt_llm_all_boxbybox(tmp_txt, sourceLang, targetLang)
        responses_all.append(tmp_response)

    return True,responses_all 


def create_mask(pil_image, box_coordinates):
    height,width  = pil_image.shape[:2]
    image_size = (height,width )
    mask = np.zeros(image_size, dtype=np.uint8)
    box_coordinates = np.array(box_coordinates, dtype=np.int32)
    cv2.fillPoly(mask, [box_coordinates], color=(255, 255, 255))
    mask=255-mask
    return mask


def PPOCR_pipline(img_path, ocr, evaluate_ocr, sourceLang, targetLang):
    trans_mode = f'{sourceLang}2{targetLang}'

    # Text Detection & Recognition
    pil_image = cv2.imread(img_path)
    result=ocr.ocr(img_path)
    
    dt_boxes = [line[0] for line in result]

    pil_image,new_dt_boxes= resize_image_boxes(pil_image,dt_boxes, max_length=768)    
    resize_image_path=img_path[:-4]+'_resize.jpg'

    cv2.imwrite(resize_image_path,pil_image )
    img_path=resize_image_path

    for i,tmp_box in enumerate(new_dt_boxes):
        new_dt_boxes[i]=enlarge_box_bigger(tmp_box)

    all_txts = [line[1][0] for line in result]
    all_text_ocr=all_txts

    if all_text_ocr==[]:
        return None

    # Translation of the detected text via API calls to an LLM
    judge,translate_responses=call_with_prompt_llm_all(all_text_ocr, trans_mode)
    
    # If translation of all image boxes at once fails, attempt translation of the boxes one by one
    if judge==False:
        judge,translate_responses=translaor_using_piple(all_text_ocr, trans_mode)

    # If both translation methods fail - write a log and return
    if judge==False:
        translation_log=img_path[:-4]+"_wrongtranslation_log.txt"
        log_file = open(translation_log, "w")
        log_file.write(str(all_text_ocr) + "\t")
        log_file.write(str(translate_responses) + "\t")        
        return None
    

    image = np.array(pil_image)
    image = image.clip(0, 255) 
    ori_image_path = img_path

    translation_log=img_path[:-4]+"translation_log.txt"
    log_file = open(translation_log, "w", encoding="utf-8")
    evaluate_log = img_path[:-4]+"evaluate_log.txt"
    evaluate_log_file = open(evaluate_log, "w", encoding="utf-8")

    for idx in range(len(new_dt_boxes)):
        boxes = new_dt_boxes[idx]
        mask = create_mask(pil_image,boxes)
        char_count_old=len(all_text_ocr[idx])
        tmp_trans = all_txts[idx]
        untranslate=False

        letterPattern = re.compile(r'[^\W\d_]', re.UNICODE)
        if re.search(letterPattern, tmp_trans):
            untranslate=True
            trans_mode='others'

        try :
            char_count_new=len(translate_responses[idx])
        except:
            return None
  
        resized_mask, whether_erase, return_box = resize_mask_returnbox_suokuan(ori_image_path,boxes,char_count_old,char_count_new,trans_mode)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        resized_masked_image=cv2.bitwise_and(image, image, mask=resized_mask)
        
        if idx==0:
            masked_image = resize_image(masked_image, max_length=768) 
            resized_masked_image= resize_image(resized_masked_image, max_length=768)

        masked_image_path=img_path[:-4]+'_masked_'+str(idx)+'.png'
        resized_masked_image_path=img_path[:-4]+'_resized_masked_'+str(idx)+'.png'
        cv2.imwrite(masked_image_path, masked_image)
        cv2.imwrite(resized_masked_image_path, resized_masked_image)
        if whether_erase==True and untranslate==False:
            InpaintImage(ori_image_path, boxes)
            erased_image_path=ori_image_path[:-4]+'_erase.png'
            ori_image_path=erased_image_path
        txts=all_txts[idx] 
        print(txts)
        log_file.write(txts + "\t")
        try:
            response=translate_responses[idx]#
        except:
            response=translaor_using_piple(txts)
        try:
            log_file.write(response + "\n")
        except:
            return None
        if response==txts:
            inpainted_image_path=img_path[:-4]+'_inpainted.png'
            cv2.imwrite(inpainted_image_path, image)
            inpainted_image=image

            evaluate_log_file.write(str(txts) + "\t")
            evaluate_log_file.write(response + "\t")
            evaluate_log_file.write(response+'\n')
        elif untranslate==True:
            inpainted_image_path=img_path[:-4]+'_inpainted.png'
            cv2.imwrite(inpainted_image_path, image)
            inpainted_image=image

            evaluate_log_file.write(str(txts) + "\t")
            evaluate_log_file.write(str(tmp_trans) + "\t")
            evaluate_log_file.write(str(tmp_trans)+'\n')
        else:
            # 3. text editing
            mode = 'text-editing'
            tmp_prompt='一张海报，上面写着' +'"'+str(response.strip('"'))+'"'
            input_data = {
                "prompt": tmp_prompt,
                "seed": 94081527,
                "draw_pos":resized_masked_image_path,
                "ori_image":ori_image_path,
            }

            try:
                results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
            except:
                translation_log=img_path[:-4]+"number_wrongtranslation_log.txt"
                log_file = open(translation_log, "w")
                log_file.write(str(all_text_ocr) + "\t")
                log_file.write(str(translate_responses) + "\t") 
                return None 
            if rtn_code >= 0:
                inpainted_image=results[0]
                inpainted_image= cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR)
                inpainted_image_path=img_path[:-4]+'_inpainted.png'
                cv2.imwrite(inpainted_image_path, inpainted_image)
                crop_image_path=image_crop(inpainted_image,return_box,inpainted_image_path,idx)
                crop_result=evaluate_ocr.ocr(crop_image_path)
                if crop_result[0]==None:
                    crop_text_write='failed'
                else:
                    crop_text= [line[1][0] for line in crop_result]
                    crop_text_write=' '.join(crop_text)

                Anytext_editing_count=0
                while response!=crop_text_write and Anytext_editing_count<5:
                    Anytext_editing_count+=1
                    random_number = random.randint(1, 10000000)
                    mode = 'text-editing'
                    tmp_prompt='一张海报，上面写着' +'"'+str(response.strip('"'))+'"'
                    input_data = {
                        "prompt": tmp_prompt,
                        "seed": random_number,
                        "draw_pos":resized_masked_image_path,
                        "ori_image":ori_image_path,
                    }
                    results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
                    inpainted_image=results[0]
                    inpainted_image= cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR)
                    inpainted_image_path_reedit=img_path[:-4]+'_inpainted_reedit.png'
                    cv2.imwrite(inpainted_image_path_reedit, inpainted_image)
                    crop_image_path=image_crop(inpainted_image,return_box,inpainted_image_path_reedit,idx)
                    crop_result=evaluate_ocr.ocr(crop_image_path)
                    if crop_result[0]==None:
                        crop_text_write='failed'
                    else:
                        crop_text= [line[1][0] for line in crop_result]
                        crop_text_write=' '.join(crop_text)
                    if Anytext_editing_count==5 or response==crop_text_write:
                        inpainted_image_path=img_path[:-4]+'_inpainted.png'
                        cv2.imwrite(inpainted_image_path, inpainted_image)              


                evaluate_log_file.write(str(txts) + "\t")
                evaluate_log_file.write(response + "\t")
                evaluate_log_file.write(crop_text_write+'\n')
                

            else:
                inpainted_image_path=img_path[:-4]+'_inpainted.png'
                cv2.imwrite(inpainted_image_path, image)
                inpainted_image=image
        ori_image_path=inpainted_image_path
        image=inpainted_image.clip(0, 255)

def main(args: argparse.Namespace):
    folder_path='/home/pepe/Documents/University year 4/Competing in machine translation/AnyTrans-extended/cs_2_all/cs2en'
    image_path_list=[]
    ocr = PaddleOCR(lang=args.sourceLang)
    evaluate_ocr = PaddleOCR(lang=args.targetLang)
    for filename in os.listdir(folder_path):
      
        if filename.endswith('.png') or filename.endswith('.jpg'):
          
            image_path = os.path.join(folder_path, filename)
            image_path_list.append(image_path)
    for i in range(250):
        path=os.path.join(folder_path,f'cs_{i+1}.png')
        if path in image_path_list:
            print(f"Processed image: {path}")

            PPOCR_pipline(path, ocr, evaluate_ocr, args.sourceLang, args.targetLang)

if __name__ == '__main__':
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    main(main_args)
            