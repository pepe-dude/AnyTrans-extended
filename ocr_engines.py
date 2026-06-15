import os
import cv2
import numpy as np

class OCRWrapper:
    def __init__(self, engine_type, lang):
        self.engine_type = engine_type
        self.lang = lang
        self.engine = None
        
        if engine_type == "paddle":
            from paddleocr import PaddleOCR

            self.engine = PaddleOCR(lang=lang)
            
        elif engine_type == "tesseract":
            import pytesseract
            self.engine = pytesseract

            # mapming program lang codes to Tesseract lang codes
            lang_map = {
                "cs": "ces",
                "en": "eng",
                "uk": "ukr",
                "german": "deu"
            }

            self.tess_lang = lang_map.get(lang, lang)

        elif engine_type == "easyocr":
            import easyocr

            lang_map = {
                "cs": "cs",
                "en": "en",
                "uk": "uk",
                "german": "de"
            }

            self.engine = easyocr.Reader([lang])

        else:
            raise ValueError(f"Unsupported OCR engine: {engine_type}")

    def ocr(self, img_path):
        if self.engine_type == "paddle":
            result = self.engine.ocr(img_path)
            if not result or result[0] is None:
                return []

            return result       
             
        elif self.engine_type == "tesseract":
            import pytesseract
            img = cv2.imread(img_path)
            if img is None:
                return []
            
            # use image_to_data to get bounding boxes and text
            data = self.engine.image_to_data(img, lang=self.tess_lang, output_type=pytesseract.Output.DICT)
            
            # group words by line to match PaddleOCR's typical output
            lines = {}
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                if conf < 0: continue # Skip blocks with no text
                text = data['text'][i].strip()
                if not text: continue
                
                # key: (block_num, par_num, line_num)
                key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
                if key not in lines:
                    lines[key] = {'left': [], 'top': [], 'right': [], 'bottom': [], 'text': []}
                
                lines[key]['left'].append(data['left'][i])
                lines[key]['top'].append(data['top'][i])
                lines[key]['right'].append(data['left'][i] + data['width'][i])
                lines[key]['bottom'].append(data['top'][i] + data['height'][i])
                lines[key]['text'].append(text)
            
            results = []
            for key in sorted(lines.keys()):
                line_data = lines[key]
                x1 = min(line_data['left'])
                y1 = min(line_data['top'])
                x2 = max(line_data['right'])
                y2 = max(line_data['bottom'])
                
                # format as [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                box = [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]
                text = " ".join(line_data['text'])
                results.append([box, [text, 0.95]]) # use a dummy confidence score
                
            return results
        
        elif self.engine_type == "easyocr":
            result = self.engine.readtext(img_path)

            if not result or result[0] is None:
                return []
            
            result = [[bbox, (text, prob)] for (bbox, text, prob) in result]

            return result
