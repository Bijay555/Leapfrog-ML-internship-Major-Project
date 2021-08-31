import logging
import numpy as np
from PIL import Image
import streamlit as st
import pytesseract.pytesseract
from src.extract_text import image_mask
from src.data_preprocess import text_preprocess
from src.prediction import Prediction_text

# the  tesseract executable in your PATH
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
custom_config = r"--oem 1 --psm 6 -c tessedit_char_whitelist= '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '"


def load_image(image_file):
    img = Image.open(image_file)
    return img

def main():
	st.title("Text Images Sentiment Analysis")
	header = st.container()
	with header:
		# Subheader
		st.subheader("Image Upload Section:")
		input_image = st.file_uploader("Upload a image file:", type = ['png','jpg','jpeg'])
		if input_image is not None:
			my_img = load_image(input_image)
			st.image(my_img, width=300)
			im = image_mask(my_img)
			# Simple image to string
			text = pytesseract.image_to_string(im, lang='eng', config=custom_config)
			st.write(text.replace('\n', ' '))

			logging.info("==== Text Preprocessing ====")
			clean_text = text_preprocess(text)
	button1 = st.button("Text Analysis")
	if button1:
		if clean_text is not None:
			model = st.container()
			
			logging.info("==== Prediction ====")
			pred = Prediction_text(clean_text)
			prediction = pred.predict_text()
			proba_predict = pred.get_prediction_probability()

			with model:
				st.success("Text Prediction")
				st.write("The sentiment is {}".format(prediction))
				st.write("Confidence score:{}".format(np.max(proba_predict)))


if __name__ == '__main__':
    main()
