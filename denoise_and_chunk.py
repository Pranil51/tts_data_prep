from tinytag import TinyTag
import os
from pydub import AudioSegment
from pydub.silence import detect_silence
from multiprocessing import Pool, cpu_count
from IPython import display as disp
from IPython.display import Audio
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import matplotlib.pyplot as plt
import glob
def process_and_denoise(file_path):
    try:
        model = pretrained.dns64().cuda()
        wav, sr = torchaudio.load(file_path)
        wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
        with torch.no_grad():
            denoised = model(wav[None])[0]
        meta = disp.Audio(denoised.data.cpu().numpy(), rate=model.sample_rate)
        with open(file_path, "wb") as f:
            f.write(meta.data)
        print("Denoising successful for", file_path)
    except Exception as e:
        print("Error processing denoising:", e)
        print("Denoising not possible for", file_path)

def create_chunks(audio,  speaker_name, output_dir, chunk_num=0,  min_segment_len=3000, max_segment_len=20000, silence_thresh=-25):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    silence = AudioSegment.silent(200)
    segment_start = 0
    while segment_start < len(audio):
        segment_end = min(segment_start + max_segment_len, len(audio))
        segment = audio[segment_start:segment_end]

        # Refine segment boundaries to nearest silence within a small buffer

        silence_ranges = detect_silence(segment, min_silence_len=200, silence_thresh=silence_thresh)
        if silence_ranges:
            last_silence = max(silence_ranges, key=lambda x: x[1]-x[0])
            silence_size = last_silence[1]-last_silence[0]
            segment_end = segment_start + (last_silence[0]+silence_size//2)
            segment = audio[segment_start:segment_end]

        # Ensure segment is at least min_segment_len
        if len(segment) >= min_segment_len:
            segment = silence + segment + silence
            segment.export(f"{output_dir}/{speaker_name}_chunk_{chunk_num}.wav", format="wav")
            print(f"Chunk {chunk_num} exported. Duration: {len(segment) / 1000} seconds")
            chunk_num += 1
        else:
            # If segment is too short, extend to next fixed length segment
            segment_end = min(segment_start + max_segment_len, len(audio))
            segment = audio[segment_start:segment_end]
            if len(segment)> min_segment_len:
                segment = silence + segment + silence
                segment.export(f"{output_dir}/{speaker_name}_chunk_{chunk_num}.wav", format="wav")
                print(f"Chunk {chunk_num} exported1. Duration: {len(segment) / 1000} seconds")
                chunk_num += 1

        segment_start = segment_end + 100  # Move to the next segment, skipping a small buffer
    return chunk_num

if __name__ == "__main__":
# Denoising and chunking
    base_dir = "Hindi"
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
            print(file)
            process_and_denoise(file)
            audio = AudioSegment.from_wav(file)
            chunk_num = create_chunks(audio,min_segment_len=min_segment_len,max_segment_len=max_segment_len,output_dir=output_dir,speaker_name=spk_name,chunk_num=chunk_num)
            print(chunk_num)

    # for spk_dir in [os.path.join(base_dir,d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,d))]:
    #     spk_files = glob.glob(pathname=spk_dir+r"/*.wav")
    #     print(spk_dir)
    #     output_dir = os.path.join("Processed",spk_dir)
    #     for file in spk_files:
    #         audio = AudioSegment.from_wav(file)
    #         chunk_num = create_chunks(audio,min_segment_len,max_segment_len,output_dir,chunk_num)
    #         print(chunk_num)