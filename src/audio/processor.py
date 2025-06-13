import librosa

class AudioProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def extract_mfcc(self):
        """Извлечение MFCC признаков из аудио"""
        signal, sr = librosa.load(self.file_path)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr)
        return mfcc.mean(axis=1)
