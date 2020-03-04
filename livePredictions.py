"""
This file can be used to try a live prediction. 
"""

from tensorflow import keras
import numpy as np
import librosa
import os
import pathlib as plb

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # set not use GPU

class livePredictions:
    """
    Main class of the application.
    """

    def __init__(self, path, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.path = path
        self.file = file

    def load_model(self):
        """
        Method to load the chosen model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        """
        self.loaded_model = keras.models.load_model(self.path)
        # return self.loaded_model.summary()

    def makepredictions(self, out_file):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=2)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        print("Prediction is: ", self.convertclasstoemotion(predictions))
        out_file.write(f"prediction: {self.convertclasstoemotion(predictions)}\n")

    @staticmethod
    def convertclasstoemotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label

# Here you can replace path and file with the path of your model and of the file 
#from the RAVDESS dataset you want to use for the prediction,
# Below, I have used a neutral file: the prediction made is neutral.

# pred = livePredictions(path='/Users/marcogdepinto/Desktop/Ravdess_V2/Emotion_Voice_Detection_Model.h5',
#                        file='/Users/marcogdepinto/Desktop/Ravdess_V2/01-01-01-01-01-01-01.wav')


def main():
    modality_list = ["01"]
    vocal_channel_list = ["01"]
    emotion_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
    emotion_intensity_list = ["01", "02"]
    statement_list = ["01", "02"]
    repetition_list = ["01", "02"]
    actor_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                  "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                  "21", "22", "23", "24"]

    result_fname = plb.Path("./Test_out/RAVDESS.txt")
    result_fname.parent.mkdir(parents=True, exist_ok=True)
    file = open(result_fname, 'w')

    audio_dir = plb.Path("./Audio_Speech_Actors_01-24/Actor_01")
    for emotion in emotion_list:
        for emotion_intensity in emotion_intensity_list:
            for statement in statement_list:
                for repetition in repetition_list:
                    if emotion == "01" and emotion_intensity == "02":
                        continue
                    sample = f"01-01-{emotion}-{emotion_intensity}-{statement}-{repetition}-01"
                    pred = livePredictions(path='./Emotion_Voice_Detection_Model.h5',
                                           file=str(audio_dir / f'{sample}.wav'))
                    print(f"sample name: {sample}; true label: {emotion}; ", end='')
                    file.write(f"sample name: {sample}; true label: {emotion}; ")

                    pred.load_model()
                    pred.makepredictions(file)

    file.close()


# script calling
if __name__ == '__main__':
    main()
