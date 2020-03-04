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

    def __init__(self, path):
        """
        Init method is used to initialize the main parameters.
        """
        self.path = path

    def load_model(self):
        """
        Method to load the chosen model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        """
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def makepredictions(self, audio_fname, out_file):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(audio_fname)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=2)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        result = self.convertclasstoemotion(predictions)

        # log output
        sample = plb.Path(audio_fname).stem
        emotion = sample.split("-")[2]
        correct = predictions[0] == int(emotion) - 1
        print(f"[Sample Name] {sample}; [Emotion]: {emotion}; [Prediction]: {result}; [{correct}]")
        out_file.write(f"[Sample Name] {sample}; [Emotion]: {emotion}; [Prediction]: {result}; [{correct}]\n")
        return correct

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
    result_fname = plb.Path("./Test_out/RAVDESS.txt")
    result_fname.parent.mkdir(parents=True, exist_ok=True)
    file = open(result_fname, 'w')

    # initialize
    pred = livePredictions(path='./Emotion_Voice_Detection_Model.h5')
    pred.load_model()

    audio_dir = plb.Path("./Audio_Speech_Actors_01-24")
    total_num = 0
    correct_num = 0
    for actor in audio_dir.iterdir():
        actor_totol = 0
        actor_correct = 0
        print(actor)
        for audio_file in actor.iterdir():
            correct = pred.makepredictions(str(audio_file), file)
            total_num += 1
            actor_totol += 1
            if correct:
                correct_num += 1
                actor_correct += 1
        actor_precision = actor_correct / actor_totol
        print(f"[{actor.stem}] precision = {actor_precision * 100:.1f}%\n")
        file.write(f"[{actor.stem}] precision = {actor_precision * 100:.1f}%\n\n")

    # compute correct rate
    precision = correct_num / total_num
    print(f"Total precision: {precision * 100:.1f}%")
    file.write(f"Total precision: {precision * 100:.1f}%")

    file.close()


# script calling
if __name__ == '__main__':
    main()
