import yaml
import joblib as jb

with open("config.yaml", "r") as stream:
    config_file_instance = yaml.safe_load(stream)
log_model = jb.load(open(config_file_instance['model_path'],'rb'))


class Prediction_text():
    def __init__(self,sent):
        self.sent = sent

    def predict_text(self):
        output = log_model.predict([self.sent])
        return output[0]

    def get_prediction_probability(self):
        output = log_model.predict_proba([self.sent])
        return output