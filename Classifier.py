import numpy as np
from PIL import Image
import requests
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, ImageClassificationPipeline
import cv2


class DogImageClassifier:

    def __init__(self):
        self.extractor = AutoFeatureExtractor.from_pretrained("roschmid/dog-races")
        self.model = AutoModelForImageClassification.from_pretrained("roschmid/dog-races")
        self.pipeline = ImageClassificationPipeline(model=self.model, feature_extractor=self.extractor)
        self.compare_dog_names = {"Border Collie": "collie/border",
                                  "Tibetan Mastiff": "mastiff/tibetan",
                                  "Chow Chow dog": "chow",
                                  "German Shepherd": "germanshepherd",
                                  "Rottweiler": "rottweiler",
                                  'Shiba Inu': "shiba",
                                  'Siberian Husky': "husky",
                                  'Golden Retriever': "retriever/golden",
                                  "Pug": "pug"
                                  }
        self.classification = None
        self.classification_score = 0.0

    def predict_image(self, image):
        return self.pipeline(image)

    def get_random_dog_image(self, breed, subbreed=""):
        if len(subbreed) > 0:
            subbreed = "/" + subbreed
        response = requests.get(url=f"https://dog.ceo/api/breed/{breed}{subbreed}/images/random")
        img = Image.open(requests.get(response.json()["message"], stream=True).raw)
        return img.resize((500, 400))

    def get_close_image(self, input_image):
        image = self.convert_cv_to_pillow(input_image)
        prediction = self.predict_image(image)[0]
        self.classification = prediction["label"]
        self.classification_score = prediction["score"]
        translated_classification = self.compare_dog_names[self.classification]
        dog_image = self.get_random_dog_image(translated_classification)
        return self.convert_pillow_to_cv(dog_image)

    def convert_cv_to_pillow(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    def convert_pillow_to_cv(self, image):
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
