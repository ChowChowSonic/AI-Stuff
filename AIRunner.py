import os
import numpy as np
import pickle
from preprocessing import Loader, Padder, LogSpectrogramExtractor, MinMaxNormaliser, Saver, PreprocessingPipeline
from Soundgenerator import soundgenerator, save_signals
from autoEncoder import VarEncoder

def load_fsdd(path, num =-1):
    x = []
    filepaths = []
    for root, _, names in os.walk(path):
        for filename in names:
            filepath = os.path.join(root, filename)
            filepaths.append(filepath)
            spectro = np.load(filepath)
            x.append(spectro)
            if num > 0 and len(x) == num:
                break
    x = np.array(x)
    x = x[..., np.newaxis]
    return x, filepaths

if __name__ == "__main__":
    Loaded = True
    DURATION = 30  # in seconds
    if Loaded == False:
        FRAME_SIZE = 512
        HOP_LENGTH = 256
        SAMPLE_RATE = 22050
        MONO = True

        SPECTROGRAMS_SAVE_DIR = "spectrograms"
        MIN_MAX_VALUES_SAVE_DIR = "fsdd"
        FILES_DIR = "recordings"

        # instantiate all objects
        loader = Loader(SAMPLE_RATE, DURATION, MONO)
        padder = Padder()
        log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
        min_max_normaliser = MinMaxNormaliser(0, 1)
        saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

        preprocessing_pipeline = PreprocessingPipeline()
        preprocessing_pipeline.loader = loader
        preprocessing_pipeline.padder = padder
        preprocessing_pipeline.extractor = log_spectrogram_extractor
        preprocessing_pipeline.normaliser = min_max_normaliser
        preprocessing_pipeline.saver = saver

        preprocessing_pipeline.process(FILES_DIR)
    Train = False
    if Train == True:
        train, _ = load_fsdd("spectrograms")
        print("Shape of training contents:",np.shape(train[0]))
        AI = VarEncoder((*np.shape(train[0]),1), (512,256,128,64,32), (3,3,3,3,3), (2,2,2,2, (2,1)), 128)
        AI.compile()
        AI.train(train[0:1], 1, 100)
        AI.save("Persona1_100:1")
    train, _ = load_fsdd("spectrograms")
    AI = VarEncoder.load("Persona32_20")
    x = AI.reconstruct(train[0:1])
    with open(".\\min_max_values.pkl", "rb") as f:
        minmaxVal = pickle.load(f)
    gen,_ = soundgenerator(AI, 256).generate(x[0:1], minmaxVal)
    save_signals(gen, "Generated")