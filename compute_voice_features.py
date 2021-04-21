"""
Batch computing of voice features for all the audios in a
tsv file. It is expected that sample ID is found at the first column,
and the path to the audio file is found at the second one.
This is the typical format in wav2letter/Flashlight and fairseq
pipelines.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
mysp = __import__("my-voice-analysis")
import soundfile as sf
import librosa
import logging
from parselmouth.praat import call, run_file
import numpy as np

logger = logging.getLogger('voice_features')
hdlr = logging.FileHandler('voice_features.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def do_compute(file):
    sound="../{}".format(file)
    sourcerun="./my-voice-analysis/myspsolution.praat"
    path="../temp/"
    print(path)
    try:
        objects= run_file(sourcerun, -20, 2, 0.3, "yes",sound,path, 80, 400, 0.01, capture_output=True)
        print(objects[0]) # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
        z1 = str(objects[1]) # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
        z2 = z1.strip().split()
        z3 = np.array(z2)
        z4 = np.array(z3)[np.newaxis]
        z5 = z4.T
        dataset = pd.DataFrame({"number_ of_syllables":z5[0,:],"number_of_pauses":z5[1,:],"rate_of_speech":z5[2,:],"articulation_rate":z5[3,:],"speaking_duration":z5[4,:],
                          "original_duration":z5[5,:],"balance":z5[6,:],"f0_mean":z5[7,:],"f0_std":z5[8,:],"f0_median":z5[9,:],"f0_min":z5[10,:],"f0_max":z5[11,:],
                          "f0_quantile25":z5[12,:],"f0_quan75":z5[13,:]})
        return dataset
    except Exception as e:  # work on python 3.x
        logger.error('Failed to upload to ftp: ' + str(e))
        print (str(e))
    return;


def measure_voice_features():

    filename = '84-121123-0002'
    path = r'./../../datasets/LibriSpeech/dev-clean/84/121123/'
    full_path = '{}{}.flac'.format(path, filename)
    print(full_path)

    # from flac to wav
    data, sr = sf.read(full_path)
    y = librosa.resample(data, 16000, 44100)
    sf.write('./temp/audio2.wav', y, 44100, 'PCM_24')


    #file_path = PurePath(full_path)
    #flac_tmp_audio_data = AudioSegment.from_file(file_path, file_path.suffix[1:])
    #flac_tmp_audio_data.export(file_path.name.replace(file_path.suffix, "") + ".wav", format="wav")
    #audio = MonoLoader(filename='./../datasets/LibriSpeech/dev-clean/84/121123/')()
    #MonoWriter(filename='test.wav')(audio)
    #os.system('ffmpeg -i {} ./output.wav'.format(full_path))

    #mysp.mysptotal('audio2', './')
    dataset = do_compute('temp/audio2.wav')
    print(dataset)
    dataset.to_csv('./temp/data.csv')



def main():
    measure_voice_features()
    return

    #parser = argparse.ArgumentParser(description="Compute voice quality features")
    #parser.add_argument("--src", default="", help="source tsv file with speech samples per row", required=True)
    #parser.add_argument("--dst", default="", help="destination tsv file with the appended voice quality measurements", required=True)
   # args = parser.parse_args()
   # print("Arguments")
   # print(args)

    df = pd.read_csv(args.src, sep='\t')

    voice_features_list = []
    for index, row in tqdm(df.iterrows()):
        path = row[1]
        assert os.path.isfile(path), f"The path \"{path}\" does not lead to a file! Check that the second column in " \
                                     f"your tsv file contains paths to audio files. "

        sound = parselmouth.Sound(path)
        voice_features = measure_voice_features(sound, args.min_f0, args.max_f0, "Hertz")

        current_information = row.tolist()
        current_information += voice_features

        voice_features_list.append(current_information)

    columns = df.columns.tolist()
    columns += ['mean_f0', 'stdev_f0', 'hnr', 'local_jitter', 'local_absolute_jitter', 'rap_jitter', 'ppq5_jitter',
                'ddp_jitter', 'local_shimmer', 'localdb_shimmer', 'apq3_shimmer', 'aqpq5_shimmer', 'apq11_shimmer',
                'dda_shimmer']
    features_df = pd.DataFrame(voice_features_list, columns=columns)

    features_df.to_csv(args.dst, sep='\t', index=None)


if __name__ == "__main__":
    main()
