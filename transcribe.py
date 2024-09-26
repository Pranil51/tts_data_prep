import azure.cognitiveservices.speech as speechsdk
import time
import csv
from pydub import AudioSegment
from pathlib import Path
import os
import glob
def transcribe_audio_with_azure(audio_path):
    try:
        sound = AudioSegment.from_file(audio_path)
        duration = sound.duration_seconds

        # Replace with your subscription key and region
        subscription_key = ''
        region = 'centralindia'

        speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
        speech_config.speech_recognition_language = "hi-IN"
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        done = False
        all_results = []

        def stop_cb(evt):
            print('CLOSING on {}'.format(evt))
            speech_recognizer.stop_continuous_recognition()
            nonlocal done
            done = True

        def handle_final_result(evt):
            all_results.append(evt.result.text)

        speech_recognizer.recognized.connect(handle_final_result)
        speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
        speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
        speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
        speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)

        speech_recognizer.start_continuous_recognition()
        while not done:
            print("Inside loop")
            time.sleep(0.5)

        print("Printing all results:", all_results)
        data = ''.join(all_results)
        total_words = len(data.split())
        print(data)
        output = [audio_path, data, duration, total_words]
        return output

        print("File written for audio:", audio_path)

    except Exception as ex:
        print(f"Error during transcription: {str(ex)}")

# Example usage
base_dir = "Processed/Hindi"
audio_file_path = "Processed\Hindi\Aditi_Rajput\Aditi_Rajput_chunk_0.wav"
transcribe_audio_with_azure(audio_file_path)
for spk_name in [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,d))]:
    spk_dir=os.path.join(base_dir,spk_name)
    spk_files = glob.glob(pathname=spk_dir+r"/*.wav")
    with open(os.path.join(spk_dir,f"metadata_{spk_name}.csv"), 'w+',encoding="utf-8") as file:
        write = csv.writer(file)
        write.writerow(["Path", "Transcript", "Duration", "Total_words"])
        for file in spk_files:
            output = transcribe_audio_with_azure(file)
            if output != None:
                write.writerow(output)
            else:
                continue
