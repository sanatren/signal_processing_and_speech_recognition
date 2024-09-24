import matplotlib.pyplot as plt
import wave 
import numpy as np

audio = wave.open("output_file1.wav","rb")

sample_freq = audio.getframerate()
n_samples = audio.getnframes()
signal_wav = audio.readframes(-1)

audio.close()

time_of_audio = n_samples/sample_freq

print(time_of_audio)

signal_array = np.frombuffer(signal_wav,dtype=np.int16)
times = np.linspace(0,time_of_audio,num=n_samples)

plt.figure(figsize=(15,5))
plt.plot(times,signal_array)
plt.title("audio signal graph")
plt.xlabel("time (s)")
plt.ylabel("signal_wave")
plt.xlim(0,time_of_audio)
plt.show()