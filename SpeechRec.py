import speech_recognition as sr
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def record(record_sec: int):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    print("녹음을 시작합니다.")
    for i in range(int(RATE/CHUNK * record_sec)):
        data = stream.read(CHUNK)
        frames.append(bytes(data))
    print("녹음을 종료합니다.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    #     wf.setnchannels(CHANNELS)
    #     wf.setsampwidth(p.get_sample_size(FORMAT))
    #     wf.setframerate(RATE)
    #     wf.writeframes(b''.join(frames))
    #     wf.close()

    return frames

def STT(record_sec: int):
    recognizer = sr.Recognizer()
    audio = record(record_sec)
    # with sr.AudioFile(audio/WAVE_FILE_NAME) as source:
    #     audio = recognizer.record(source)
    try:
        txt = recognizer.recognize_google(audio_data=audio, language='ko-KR')
    except sr.UnknownValueError:
        print("언어가 인지되지 않았습니다")
        return None

    return txt
