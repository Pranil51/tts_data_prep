from yt_dlp.postprocessor import FFmpegPostProcessor
FFmpegPostProcessor._ffmpeg_location.set(R'C:\FFmpeg\bin\ffmpeg.exe')
import os
import shutil
def to_seconds(timestamp):
    min, sec = list(map(int,timestamp.split(':')))
    return min*60 + sec
file = open("hf_token.txt","r")
from extract_from_multispeaker import *
from denoise_and_chunk import *
import pandas as pd
if __name__=="__main__":
    # Extracting single speaker utterances
    # df = pd.read_csv("speaker_data - xtts.csv")
    df = pd.read_excel('speaker_data.xlsx', sheet_name=1)
    df = df.drop(columns=['Note','Other Speaker'])
    n=0
    if not os.path.exists('Utterances'):
        os.mkdir('Utterances')
    for speaker, l1, start1, end1, l2, start2, end2, l3,start3, end3, l4, start4, end4, l5, start5, end5, language in df.itertuples(index=False):
        try:
            links = list(filter(lambda x: isinstance(x,str),[l1,l2,l3,l4,l5]))
            ends = list(map(to_seconds,filter(lambda x: isinstance(x,str),[end1,end2,end3,end4,end5])))
            starts = list(map(to_seconds,filter(lambda x: isinstance(x,str),[start1,start2,start3,start4,start5])))
            speaker = '_'.join(speaker.split(' '))
            output_dir=os.path.join('Utterances',language)
            audio_paths = [Audio_from_YT(link, speaker + f"{i}", output_dir) for i, link in enumerate(links)]
            extract_utterance(audio_paths,starts,ends,
                                file.read(),
                                output_dir, speaker)
        except RuntimeError as e:
            print(e, "failed to process speaker data. Continuing for the next speaker.")
            continue
        if n==3:
            break
        n+=1
    # chunking the extracted single utterances.
    for base_dir in os.listdir("Utterances"):
        print(base_dir)
        max_segment_len = 1000*20 - 400
        min_segment_len = 1000*3
        target_dir = os.path.join("Processed",base_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for spk_name in [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,d))]:
            chunk_num = 0
            spk_dir=os.path.join(base_dir,spk_name)
            print(spk_dir)
            spk_files = glob.glob(pathname=spk_dir+r"/*.wav")
            output_dir = os.path.join("Processed",spk_dir)
            for file in spk_files:
                process_and_denoise(file)
                audio = AudioSegment.from_wav(file)
                chunk_num = create_chunks(audio,min_segment_len=min_segment_len,max_segment_len=max_segment_len,output_dir=output_dir,speaker_name=spk_name,chunk_num=chunk_num)
    shutil.rmtree('Utterances')
