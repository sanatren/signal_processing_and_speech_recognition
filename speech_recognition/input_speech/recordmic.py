import wave
import pyaudio

frames_pr_buffer = 3200
Format = pyaudio.paInt16
Channels = 1
Rate = 16000

p = pyaudio.PyAudio()

stream = p.open(
    format = Format,
    channels = Channels,
    rate = Rate,
    input = True,
    frames_per_buffer = frames_pr_buffer
)

print("start recording")
seconds = 3
frames = []
for i in range (0,int(Rate/frames_pr_buffer*seconds)):
    data = stream.read(frames_pr_buffer)
    frames.append(data)

stream.stop_stream()

stream.close()
p.terminate()

#creating the output file for audio input
obj = wave.open("myName.wav","wb")
obj.setnchannels(Channels)
obj.setsampwidth(p.get_sample_size(Format))
obj.setframerate(Rate)
obj.writeframes(b"".join(frames))
obj.close()

