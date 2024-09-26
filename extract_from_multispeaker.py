from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import numpy as np
import torch
import subprocess
import os
from os import path
import pandas as pd
import librosa
import soundfile as sf
from numpy.linalg import norm
from umap import UMAP
from resemblyzer import preprocess_wav, VoiceEncoder
import csv
def Audio_from_YT(link, speaker, output_dir, format="wav"):
    pwd = os.getcwd()
    out_path=path.join(pwd,output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(out_path)
    out_name = path.join(output_dir,f"{speaker}_org.{format}")
    try:
        subprocess.run(f"yt-dlp -x --audio-format wav -o {out_name} {link}", shell=True)
        print(f'Audio file from YouTube is downloaded successfully.{link}')
        return out_name
    except Exception as e:
        print(e, link)
        return 
encoder = VoiceEncoder()

def Diarize(audio_path, speaker_name, auth_token, output_dir):
    ## load model and diarize
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token)
    pipeline.to(torch.device("cuda"))
    with ProgressHook() as hook:
        diarization = pipeline(audio_path, hook=hook)

    dia_output_file = f'{output_dir}/{speaker_name}.csv'
    with open(dia_output_file, 'w') as f1:
        writer = csv.writer(f1)
        writer.writerow(["speaker", "start", "end", "duration"])
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            writer.writerow([speaker, turn.start, turn.end, turn.end-turn.start])
    return dia_output_file
def extract_utterance(audio_paths, ref_starts, ref_ends, auth_token, output_dir, speaker_name):
    ## pyannote diarization
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    def Identify_Speaker(speakers_sample, ref_utterance, source_sr):
        speakers_sample["ref_utterance"] = ref_utterance
        spks = list(speakers_sample.keys())
        spk_wavs = [preprocess_wav(wav, source_sr) for wav in speakers_sample.values()]
        embeds = np.array([encoder.embed_utterance(wav) for wav in spk_wavs])
        similarity = np.inner(embeds, embeds)[-1,:-1]
        mask = similarity == np.max(similarity)
        most_sim_speaker = np.array(spks[:-1])[mask][0]
        return most_sim_speaker
    def Utterences_Similarity(segment_wavs, ref_utterance, source_sr):
        segment_wavs["ref_utterance"] = ref_utterance
        wavs = [preprocess_wav(wav, source_sr) for wav in segment_wavs.values()]
        XY = np.array(list(map(encoder.embed_utterance, wavs)))
        print("shape of XY: ",XY.shape)
        reducer = UMAP(n_components=10,init='random')
        if XY.shape[0]!=2:
            XY_reduced = reducer.fit_transform(XY)
        else:
            XY_reduced = XY
        X_reduced = XY_reduced[:-1,:]
        Y_reduced = XY_reduced[-1,:]
        print("Reduced to shape: ", X_reduced.shape)
        print(Y_reduced.shape)
        segment_wavs.pop("ref_utterance")
        cosine = [np.dot(X_r, Y_reduced) / (norm(X_r) * norm(Y_reduced)) for X_r in X_reduced]
        similarity = {segment: sim for segment, sim in zip(segment_wavs.keys(), cosine)}
        return similarity
    all_data = pd.DataFrame(columns=['speaker','start', 'end', 'duration','segment', 'similarity', "audio_num" ])
    all_segments = {}
    n = 1
    audio_num = 1
    for audio_path, ref_start, ref_end in zip(audio_paths, ref_starts, ref_ends):
        try:
            dia_output_file = Diarize(audio_path,speaker_name, auth_token, output_dir)
            data = pd.read_csv(dia_output_file)
        except Exception as e:
            print(f'Invalid Audio Path {audio_path} for {speaker_name}.', e)
            continue

        ## Loading audio file and defining main-speaker using reference segment in it.
        audio, source_sr = librosa.load(audio_path)
        ref_utterance = audio[ref_start * source_sr: ref_end * source_sr]
        sample_data = data.loc[data.groupby("speaker")["duration"].idxmax()]
        speakers_sample = {speaker: audio[int(start*source_sr): int(end*source_sr)] for speaker, start, end, _ in sample_data.itertuples(index=False)}
        main_speaker = Identify_Speaker(speakers_sample, ref_utterance, source_sr)
        print("main speaker: ",main_speaker)
        ## drop edge segments for main speaker with duration less than 2 sec. (small edge segments prone to overlapping)
        drop_rows = (
                ((data['speaker'] == main_speaker) & (data['duration'] < 2) & (
                            data['speaker'] != data['speaker'].shift())) |
                ((data['speaker'] == main_speaker) & (data['duration'] < 2) & (
                            data['speaker'] != data['speaker'].shift(-1)))
        )
        # Drop the marked rows
        data = data[~drop_rows]

        ## assign group for each speaker block
        data['group'] = ((data['speaker'] != data['speaker'].shift()) | (data['start'] - data['end'].shift() > 1) | (data['duration']>30)).cumsum()
        data = data.groupby(['group']).agg({'speaker': 'first', 'start': 'first', 'end': 'last'}).reset_index(drop=True)
        data['duration'] = data['end'] - data['start']

        ## adjusting start, end values for main-speaker segments to remove overlapping
        l = len(data)
        if data['speaker'][0] == main_speaker and data['end'][0] > data['start'][1]:
            data['end'][0] = data['start'][1]
        if data['speaker'][l - 1] == main_speaker and data['start'][l - 1] < data['end'][l - 2]:
            data['start'][l - 1] = data['end'][l-2]
        for i in range(1, l - 1):
            if data['speaker'][i] == main_speaker:
                if data['start'][i] < data['end'][i - 1]:
                    data['start'][i] = data['end'][i - 1]
                if data['end'][i] > data['start'][i + 1]:
                    data['end'][i] = data['start'][i + 1]
        data = data[data['start'] < data['end']]
        data['duration'] = data['end'] - data['start']
        data = data[data['duration'] > 3]
        data = data[data['speaker'] == main_speaker]
        data = data.round(2)

        ## additional filtering out remaining overlapped segments using clustering method
        segment_wavs = {}
        for spk, start, end, duration in data.itertuples(index=False):
            start = int(start * source_sr)
            end = int(end * source_sr)
            segment = audio[start:end]
            segment_name = f"{speaker_name}_seg_{n}"
            segment_wavs[segment_name] = segment
            n += 1
        similarity = Utterences_Similarity(segment_wavs, ref_utterance, source_sr)
        data['segment'] = list(similarity.keys())
        data['similarity'] = list(similarity.values())
        data['audio_num'] = audio_num
        filtered_segments = {
            segment: wav for segment, wav in segment_wavs.items()
            if similarity[segment] >= 0.986
        }
        print("total segments filtered out: ", len(segment_wavs)-len(filtered_segments))
        all_segments.update(filtered_segments)
        all_data = all_data._append(data, ignore_index=True)
        os.remove(audio_path)
        audio_num+=1
    ## export speaker segments
    pwd = os.getcwd()

    speaker_dir = os.path.join(pwd, output_dir, speaker_name)
    if not os.path.exists(speaker_dir):
        os.mkdir(speaker_dir)
    for f_segment, f_wav in all_segments.items():
        sf.write(f"{speaker_dir}/{f_segment}.wav", f_wav, source_sr)
    total_duration = all_data['duration'].sum()
    print(f"{len(all_segments)} segments exported for {speaker_name}. total duration: {total_duration}")
    all_data.to_csv(f"{speaker_dir}/metadata_{speaker_name}.csv", index = False)
