import yaml
import joblib as jb
import logging
import numpy as np
from PIL import Image
import streamlit as st
import pytesseract.pytesseract
from src.extract_text import image_mask
from src.data_preprocess import text_preprocess

# the  tesseract executable in your PATH
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
custom_config = r"--oem 1 --psm 6 -c tessedit_char_whitelist= '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '"

with open("config.yaml", "r") as stream:
    config_file_instance = yaml.safe_load(stream)
log_model = jb.load(open(config_file_instance['model_path'],'rb'))

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_text(sen: str):
	'''function: to predict the sentiment class of the given text

			parameters
			sen: string

			returns
			sentiment class: string
	'''
	output = log_model.predict([sen])
	return output[0]

def get_prediction_probability(sen: str):
	'''function: to predict the sentiment class probability score of the given text

			parameters
			sen: string

			returns
			sentiment class: int
	'''
	output = log_model.predict_proba([sen])
	return output

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
            prediction = predict_text(clean_text)
            proba_predict = get_prediction_probability(clean_text)

            with model:
                st.success("Text Prediction")
                st.write("The sentiment is {}".format(prediction))
                st.write("Confidence score:{}".format(np.max(proba_predict)))



if __name__ == '__main__':
    main()
