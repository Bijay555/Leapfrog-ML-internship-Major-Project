# Leapfrog-ML-internship-Major-Project

### Title: Image based Sentiment Analysis

### Dataset:
The dataset chosen to train a text classification model is Twitter Text Emotion Analysis Dataset. The dataset is taken from Data.World. In a variation on the popular task of sentiment analysis, this dataset contains labels for the emotional content (such as happiness, sadness, and anger) of texts. There are around 40000 samples in the dataset.
Link: https://data.world/crowdflower/sentiment-analysis-in-text

The dataset has 4 columns:

tweets id : unique id for each tweets
sentiments: the text emotion . There are 7 different emotions taken as label classes in the model. 
```
+ joy         
+ sadness      
+ fear                 
+ surprise     
+ neutral      
+ disgust       
+ shame         
```
content: tweets 
author: user who wrote those tweets


Scope:
Many times that people love to share their thoughts and feelings to their connections on social medias like facebook, twitter, instagram etc. There are various ways to share your thoughts on these platforms. One of the ways is with text images/ quote images. 
So , the project is about finding the sentiments through text images.


+ Data Analysis and Visualization on twitter raw data
+ Model Building and Evaluation
+ Use of open source OCR library ie Pytesseract for image preprocessing
+ Model trained with only English language.
+ Model trained with only black and white text images


The project consists of use of OCR Open Source Library Pytesseract. Pytesseract is a wrapper for Tesseract-OCR Engine. It is also useful as a stand-alone invocation script to tesseract, as it can read all image types supported by the Pillow and Leptonica imaging libraries, including jpeg, png, gif, bmp, tiff, and others.
Simple steps for tesseract installation in windows.
```
Download tesseract exe from https://github.com/UB-Mannheim/tesseract/wiki
Install this exe in C:\Program Files (x86)\Tesseract- OCR
Open virtual machine command prompt in windows or anaconda prompt.
Run pip install  tesseract
To test if tesseract is installed type in python prompt:
import pytesseract
print(pytesseract)
```

### Project Structure
```
D:.
│   .gitignore
│   app.py
│   config.yaml
│   Pipfile
│   Pipfile.lock
│   README.md
│
├───data
│       tweet_emotions.csv
│
├───model
│       text_model.pkl
│
├───Notebook
│       Data_Preprocessing.ipynb
│       Model_Building.ipynb
│
└───src
        data_preprocess.py
        extract_text.py
        prediction.py
        
   ```

Folders/files Description:

1. src
+ Contains the data_preprocess.py which takes the input as string or text and removes all stopwords and punctuations and return clean text
+ Contains the extract_text.py that takes image as input and perform cv2 operations. i.e bilateral filter.( bilateral filter separates foreground with background which makes it easier to extract images.
+ Contains the prediction.py which takes the preprocessed testing text and predict with the model loaded. 

2. app.py
the app.py file that calls the other functions py file to perform image to text processing and the extracted text goes through data preprocessing and is finally predicted from the model.

4. Config.yaml
has model paths 

5. model
The model folder consist of pickle ML model. The pickle operation is used to serialize your machine learning algorithms and save the serialized format to a file

6. Pipline and Pipfile.lock
The pipenv environment locks the packages used in the project. Pipenv is a packaging tool for Python that solves some common problems associated with the typical workflow using pip , virtualenv , and the good old requirements. txt 

How to run this repository:
Pre-requisits: Install pipenv(sudo apt-get pipenv)

Step 1: Setup
Clone the repository.
pipenv shell
setup the folder

Step 2: Run for py file
Run "python app.py"

Output:
Streamlit Gui Output of sentiment class with confidence score

Image Output:
![alt text](https://github.com/Bijay555/Leapfrog-ML-internship-Major-Project/blob/dev/images/output_images/Screenshot%20(439).png)
----------------------------------------------------------------------------------------------------------------------------------------------------------
![alt text](https://github.com/Bijay555/Leapfrog-ML-internship-Major-Project/blob/dev/images/output_images/Screenshot%20(440).png)
