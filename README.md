# Speech Recognition Repository

This project implements a comprehensive speech recognition pipeline using advanced signal processing techniques and deep learning. The system processes audio signals, extracts relevant features, and employs neural networks to recognize and classify speech patterns.
Key Features

Audio Recording and Processing: Capture and manipulate audio signals using wave and pyaudio libraries
Time-Domain Feature Extraction: Calculate amplitude envelope, RMS energy, and zero-crossing rate for temporal analysis
Frequency-Domain Analysis: Apply Fourier Transform to convert signals from time domain to frequency domain
Spectrogram Generation: Create visual representations of the spectrum of frequencies in audio signals
Mel-Spectrogram Extraction: Generate perceptually relevant audio features using the Mel scale
PyTorch Integration: Implement deep learning models for audio classification and pattern recognition

## Repository Structure

1-input_speech: Basic audio manipulation including reading, writing, and visualizing WAV files
2-Audio&DL through Pytorch: Integration of PyTorch for deep learning on audio signals
3-Extracting time domain features: Scripts for temporal feature extraction
4-Extracting RMSenergy and Zero crossing rate: Implementation of RMS energy and ZCR calculations
5-fourier transform for waves: Basic implementation of the Fourier Transform for wave analysis
6-fourier transform with complex numbers: Advanced Fourier analysis using complex number representation
7-extracting fourier transform from audios: Application of Fourier techniques to real audio signals
8-Extracting spectrograms from audio: Generation of spectrograms for feature extraction
9-extracting-mel-spectograms: Implementation of Mel-scale spectrogram extraction

### Technologies Used

Python
PyAudio
NumPy
Matplotlib
PyTorch
Librosa (for audio processing)
Wave (for WAV file manipulation
