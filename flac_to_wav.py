import os
import argparse
import soundfile as sf
import librosa

PATH_AUDIO_TEMP = './temp/audio.wav'
PATH_DIR_TEMP = './temp'


def convert(src_path, dst_path=PATH_AUDIO_TEMP):
    if not dst_path:
        dst_path = PATH_AUDIO_TEMP

    print("Convert: {} -----> {}".format(src_path, dst_path))
    data, sr = sf.read(src_path)
    y = librosa.resample(data, 16000, 44100)
    sf.write(dst_path, y, 44100, 'PCM_24')
    return dst_path


def setup():

    if not os.path.exists(PATH_DIR_TEMP):
        os.makedirs(PATH_DIR_TEMP)


def main():

    parser = argparse.ArgumentParser(description="Compute voice features")
    parser.add_argument("--src", default="", help="source tsv file with speech samples per row", required=True)
    parser.add_argument("--dst", default="", help="destination tsv file with the appended measurements", required=False)
    args = parser.parse_args()

    setup()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()