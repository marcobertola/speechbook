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
import logging
from parselmouth.praat import call, run_file
import numpy as np
import flac_to_wav as f2w
import pathlib
#__import__("my-voice-analysis")
PATH_FILE_DIR = pathlib.Path(__file__).parent.absolute()
PATH_AUDIO_TEMP = '{}/temp/audio.wav'.format(PATH_FILE_DIR)
PATH_TEMP_DIR = '{}/temp'.format(PATH_FILE_DIR)
PATH_DEFAULT_DATA_DST = "{}/temp/data.csv".format(PATH_FILE_DIR)

print("PATH_FILE_DIR: {}".format(PATH_FILE_DIR))
print("PATH_AUDIO_TEMP: {}".format(PATH_AUDIO_TEMP))
print("PATH_TEMP_DIR: {}".format(PATH_TEMP_DIR))


def do_compute(file, logger):

    sound = "{}/..{}".format(PATH_FILE_DIR, file)
    source_run = "{}/my-voice-analysis/myspsolution.praat".format(PATH_FILE_DIR)
    path = "{}/../temp/".format(PATH_FILE_DIR)

    print("File: {}".format(file))
    print("Sound: {}".format(sound))
    print("Path: {}".format(path))
    try:
        objects = run_file(source_run, -20, 2, 0.3, "yes", sound, path, 80, 400, 0.01, capture_output=True)
        z1 = str(objects[1])  # This will print the info from the textgrid object
        z2 = z1.strip().split()
        z3 = np.array(z2)
        z4 = np.array(z3)[np.newaxis]
        z5 = z4.T
        dataset = pd.DataFrame({"number_ of_syllables": z5[0, :],
                                "number_of_pauses": z5[1, :],
                                "rate_of_speech": z5[2, :],
                                "articulation_rate": z5[3, :],
                                "speaking_duration": z5[4, :],
                                "original_duration": z5[5, :],
                                "balance": z5[6, :],
                                "f0_mean": z5[7, :],
                                "f0_std": z5[8, :],
                                "f0_median": z5[9, :],
                                "f0_min": z5[10, :],
                                "f0_max": z5[11, :],
                                "f0_quantile25": z5[12, :],
                                "f0_quan75": z5[13, :]})
        return dataset
    except Exception as e:
        logger.error('Failed to upload to ftp: ' + str(e))
        print(str(e))
    return


def measure_voice_features(df, logger):

    frames = list()
    total_rows = len(df)
    for index, row in tqdm(df.iterrows()):
        file_path = row[1]
        print(r"Process file: {}\{} \t {}".format(index, total_rows, file_path))
        assert os.path.isfile(file_path), f"The path \"{path}\" does not lead to a file! Check that the second column" \
                                          f"in your tsv file contains paths to audio files. "
        if file_path.endswith('.flac'):
            file_path = f2w.convert(file_path)

        frames.append(do_compute(file_path, logger))
        break

    dataset = pd.concat(frames)
    dataset = dataset.reset_index()
    return dataset


def setup():

    if not os.path.exists(PATH_TEMP_DIR):
        os.makedirs(PATH_TEMP_DIR)

    logger = logging.getLogger('voice_features')
    hdlr = logging.FileHandler('voice_features.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)

    return logger


def main():

    parser = argparse.ArgumentParser(description="Compute voice features")
    parser.add_argument("--src", default="", help="source tsv file with speech samples per row", required=True)
    parser.add_argument("--dst", default="", help="destination tsv file with the appended measurements", required=False)
    args = parser.parse_args()
    if args.dst:
        destination = args.dst
    else:
        destination = PATH_DEFAULT_DATA_DST

    logger = setup()

    # read file
    df = pd.read_csv(args.src, sep='\t')

    # computer features
    features_df = measure_voice_features(df, logger)

    # merge and store info
    final_df = pd.concat([df, features_df], axis=1)
    final_df.to_csv(destination, sep='\t', index=None)


if __name__ == "__main__":
    main()
