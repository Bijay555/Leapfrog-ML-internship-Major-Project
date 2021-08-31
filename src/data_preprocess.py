

def text_preprocess(text: str):
  ''' function: to clean the noises from the text data and preprocess the data

        parameters:
        text : str

        return:
        text : str
  '''
  text = text.replace('\n', ' ')
  # convert all words into lowercase
  text = text.lower()

  # Single character removal
  text = text.replace("\s+[a-zA-Z]\s+", ' ')

  #remove the punctuation and grammatical syntax from the text
  text = text.replace('[?|!|\'|\[|\]|"|#|=]',' ')
  text = text.replace('[.|,|)|(|\|/]',' ')

  # Removing multiple spaces
  text = text.replace('\s+', ' ')

  return text