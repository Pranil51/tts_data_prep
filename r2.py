from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence

def create_chunks(audio, min_segment_len, max_segment_len, silence_thresh=-25, output_dir="test"):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    silence = AudioSegment.silent(200)
    segment_start = 0
    chunk_num = 0
    while segment_start < len(audio):
        segment_end = min(segment_start + max_segment_len, len(audio))
        segment = audio[segment_start:segment_end]

        # Refine segment boundaries to nearest silence within a small buffer
        silence_ranges = detect_silence(segment, min_silence_len=200, silence_thresh=silence_thresh)
        print(silence_ranges,len(audio))
        if silence_ranges:
            last_silence = max(silence_ranges, key=lambda x: x[1]-x[0])
            print(last_silence)
 
            segment_end = segment_start + last_silence[0] +100
            segment = audio[segment_start:segment_end]

        # Ensure segment is at least min_segment_len
        if len(segment) >= min_segment_len:
            segment = silence + segment + silence
            segment.export(f"{output_dir}/chunk_{chunk_num}.wav", format="wav")
            print(f"Chunk {chunk_num} exported. Duration: {len(segment) / 1000} seconds")
            chunk_num += 1
        else:
            # If segment is too short, extend to next fixed length segment
            segment_end = min(segment_start + max_segment_len, len(audio))
            segment = audio[segment_start:segment_end]
            if len(segment)> min_segment_len:
                segment = silence + segment + silence
                segment.export(f"{output_dir}/chunk_{chunk_num}.wav", format="wav")
                print(f"Chunk {chunk_num} exported1. Duration: {len(segment) / 1000} seconds")
                chunk_num += 1

        segment_start = segment_end   # Move to the next segment, skipping a small buffer

    return chunk_num

# Example usage
audio = AudioSegment.from_wav('Hindi\Arpita_Arya\Arpita_Arya_seg_3.wav')
min_segment_len = 3000  # 3 seconds
max_segment_len = 20000 - 400 # 20 seconds
create_chunks(audio, min_segment_len, max_segment_len)
