import librosa, pickle
import os, numpy as np, soundfile as sf
from preprocessing import MinMaxNormaliser
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

def save_signals(signals, directory, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(directory, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)

class soundgenerator: 
    def __init__(self,vae, hop_len):
        self.vae = vae
        self.hoplen = hop_len
        self._min_max_norm = MinMaxNormaliser(0, 1)
    
    def generate(self, spectrogram, min_max_values):
        generated_spectros, latent_reps = self.vae.reconstruct(spectrogram)
        signals = self.revert_to_audio(generated_spectros, min_max_values)
        return signals, latent_reps

    def revert_to_audio(self, spectros, minmax):
        signals = []
        for gram, minmaxvalue in zip(spectros, minmax.keys()):
            # Reshape the log spectrogram
            log_spectro = gram[:,:,0]
            #Apply denormalization
            denorm_log_spec = self._min_max_norm.denormalise(log_spectro, minmax[minmaxvalue]["min"], minmax[minmaxvalue]["max"])
            #Linearize the spectrogram
            spec = librosa.core.db_to_amplitude(log_spectro)
            signal = librosa.core.istft(spec, hop_length=self.hoplen)
            signals.append(signal)
        return signals

if __name__ == "__main__":
    #Init generator, load spectros, sample them
    vae = VarEncoder.load("Persona32_20")
    sound_gen = soundgenerator(vae, 256)
    with open(".\\min_max_values.pkl", "rb") as f:
        minmaxVal = pickle.load(f)
    specs, file_paths = load_fsdd("spectrograms")

    signals, _ = sound_gen.generate([specs[0]], [minmaxVal[0]])
    #OGs = sound_gen.revert_to_audio(specs[0], minmaxVal[0])

    save_signals(signals, "")
    #save_signals(OGs, "")