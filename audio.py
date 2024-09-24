import wave

obj = wave.open("input_file.wav","rb")

print("no. of channels " ,obj.getnchannels())
print("no of frames ", obj.getnframes())
print("sample_width ",obj.getsampwidth())
print("framerate ",obj.getframerate())
print( "parameters ",obj.getparams())

time_of_audio = obj.getnframes()/obj.getframerate()
print(time_of_audio)

frames = obj.readframes(-1)
print(type(frames))
print(len(frames)/2) #since the chaneels are two 

obj.close()

# Create a new WAV file for writing
obj_new = wave.open("output_file1.wav", "wb")

# Set parameters for the new file
obj_new.setnchannels(1)  
obj_new.setsampwidth(2)  
obj_new.setframerate(44100)  #Ensure this is set appropriately

# Write the frames to the new file
obj_new.writeframes(frames)

# Close the new file
obj_new.close()



