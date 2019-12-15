import webrtcvad
import pyaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import tensorflow as tf

audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 480
RECORD_SECONDS = 8

vad = webrtcvad.Vad()
vad.set_mode(3)

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

frames = []
frameCount = 0

while frameCount < 5:
    data = stream.read(CHUNK)

    if vad.is_speech(data, RATE):
        frameCount+=1;
    else:
        frameCount = 0;

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    if vad.is_speech(data, RATE):
        frames.append(data)
        


frames = np.array([ np.frombuffer(frame, 'int16').astype(np.float32) for frame in frames ])
print(frames.shape)
frames = frames.flatten()


params = {'dim': (257, None, 1),
          'nfft': 512,
          'spec_len': 250,
          'win_length': 400,
          'hop_length': 160,
          'n_classes': 5994,
          'sampling_rate': 16000,
          'normalize': True,
          }

## extend wav
extended_wav = np.append(frames,frames[::-1])
linear_spect = librosa.stft(extended_wav, n_fft=params['nfft'], win_length=params['win_length'], hop_length=params['hop_length']).T
mag, _ = librosa.magphase(linear_spect)  # magnitude
print('shappper', mag.shape)
mag_T = mag.T
freq, time = mag_T.shape
spec_mag = mag_T
print('shappper', spec_mag)
# preprocessing, subtract mean, divided by time-wise var
mu = np.mean(spec_mag, 0, keepdims=True)
std = np.std(spec_mag, 0, keepdims=True)
testsample = (spec_mag - mu) / (std + 1e-5)
testsample = np.expand_dims(np.expand_dims(testsample,0),-1)
print('shappper', testsample.shape)
model = tf.keras.models.load_model('./voice_model.tf')

against = np.load('./predictor.npy')
output = model.predict(testsample)
# np.save('./2', output)
myself = np.load('./2.npy')

print(output, output.shape)

from scipy.spatial import distance
print('cosine', distance.cosine(output[0], myself[0]))
print('norm', np.linalg.norm(output[0]-myself[0]))
print('eucl', distance.euclidean(output[0],myself[0]))
for item in against:
    print(item.shape, output.shape)
    print('cosine', distance.cosine(output[0], item[0]))
    print('norm', np.linalg.norm(output[0]-item[0]))
    print('eucl', distance.euclidean(output[0],item[0]))
    print('>>>>>>>>>>>>>')

