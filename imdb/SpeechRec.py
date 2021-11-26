import speech_recognition as sr
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
OUTPUT_FILENAME = "/textClassifier/imdb/audio/output.wav"

def record(record_sec: int):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    print("녹음을 시작합니다.")
    for i in range(int(RATE/CHUNK * record_sec)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("녹음을 종료합니다.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    return b''.join(frames)

def STT(record_sec: int):
    record(record_sec)
    recognizer = sr.Recognizer()
    with sr.AudioFile(OUTPUT_FILENAME) as source:
        audio = recognizer.record(source)
    try:
        txt = recognizer.recognize_google(audio_data=audio, language='en-UR')
    except sr.UnknownValueError:
        print("언어(영어)가 인지되지 않았습니다")
        return None
    print("인지된 문자 : "+txt)

    return txt
