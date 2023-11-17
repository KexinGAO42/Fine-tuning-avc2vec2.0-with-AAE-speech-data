import textgrid
from pydub import AudioSegment
import os
import random
import json
from scipy.io import wavfile
import noisereduce as nr


'''
1. Read a TextGrid object from a file.
'''

tg = textgrid.TextGrid.fromFile('../raw_data/ROC_se0_ag1_f_02_1.TextGrid')

# Goal: Create a list of [(start_time, end_time), text] for each people
# Skip any intervals that has empty text
all_data = {}  # people_name => list of [((start_time, end_time), text)]
for people_idx in range(len(tg)):
    print("Entering people {}: {}".format(people_idx, tg[people_idx].name))
    print("Start time {}, end time {}, num of intervals {}".format(tg[people_idx].minTime,
                                                                   tg[people_idx].maxTime,
                                                                   len(tg[people_idx])))

    all_data[tg[people_idx].name] = []

    for interval_idx in range(len(tg[people_idx])):
        interval = tg[people_idx][interval_idx]
        if interval.mark:
            one_data = ((interval.minTime, interval.maxTime), interval.mark)
            all_data[tg[people_idx].name].append(one_data)
    print()


'''
2. Data segmentation and clean
'''

print("Begin audio segmentation...")

# load data
rate, data = wavfile.read("../raw_data/ROC_se0_ag1_f_02_1.wav")

os.chdir("../processed_data")

# noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("ROC_se0_ag1_f_02_1_denoise.wav", rate, reduced_noise)

bigAudio = AudioSegment.from_wav("ROC_se0_ag1_f_02_1_denoise.wav")

CHOSEN_PEOPLE = "ROC_se0_ag1_f_02"

MIN_SEG_LENGTH = 10

if not os.path.exists("Split"):
    os.mkdir("Split")
os.chdir("Split")

all_file_prefix = []
for people_name, people_data_list in all_data.items():

    if CHOSEN_PEOPLE is not None and CHOSEN_PEOPLE != people_name:
        continue

    print("Dealing with people {}...".format(people_name))
    if not os.path.exists(people_name):
        os.mkdir(people_name)

    local_start = None  # start of current interval
    local_len = None   # current interval length
    local_text = None  # current interval text
    local_wav = None   # current interval wav segment
    for one_data in people_data_list:
        start_time = one_data[0][0]
        end_time = one_data[0][1]
        text = str(one_data[1])

        #########
        punc_list = [',', '.', '!', '>', '/', '&', '-', ':', ';', '@', '...', '\'', '\"']
        for c in punc_list:
            text = text.replace(c, '')

        if '[' in text or '/' in text or '<' in text or '(' in text:
            continue
        if any(char.isdigit() for char in text):
            continue

        text = text.upper()
        #########

        if local_start is None:
            local_start = start_time
            local_len = 0
            local_text = ""
            local_wav = bigAudio[start_time * 1000: start_time * 1000 + 1] # I hope this 1ms repeat won't hurt anyone...

        local_len += end_time - start_time
        local_text += " " + text

        local_wav += bigAudio[start_time * 1000:end_time * 1000]

        if local_len >= MIN_SEG_LENGTH:
            one_wav_path = "{}_{}.wav".format(local_start, local_len)
            txt_path = "{}_{}.txt".format(local_start, local_len)

            local_wav.export(os.path.join(people_name, one_wav_path), format="wav")
            with open(os.path.join(people_name, txt_path), 'w') as f:
                f.write(local_text)

            all_file_prefix.append(os.path.join(people_name, "{}_{}".format(local_start, local_len)))

            # Reset everything
            local_start = None
            local_len = None
            local_text = None
            local_wav = None


'''3. Create json for train/val/test (80/5/15) The fine-tuning package we will use prefer reading in json file, 
so we do this conversion during the data preprocessing phase'''

print("Total number of data: {}".format(len(all_file_prefix)))
num_data = len(all_file_prefix)

random.seed(950601)
random.shuffle(all_file_prefix)

train_list = all_file_prefix[:int(num_data * 0.8)]
val_list = all_file_prefix[int(num_data * 0.8): int(num_data * 0.85)]
test_list = all_file_prefix[int(num_data * 0.85):]

print("Number of train data: {}".format(len(train_list)))
print("Number of val data: {}".format(len(val_list)))
print("Number of test data: {}".format(len(test_list)))

for job_name, job_list in [("train", train_list), ("val", val_list), ("test", test_list)]:

    with open(job_name+".json", 'w') as jf:

        all_test_list = []
        for one_path in job_list:
            text = None
            with open(one_path + ".txt", 'r') as f:
                text = f.readlines()[0]

            one_dict = {"file": one_path + ".wav", "text": text}
            all_test_list.append(text)
            jf.write(json.dumps(one_dict) + "\n")

        # Write test sentences for later comparison
        if job_name == "test":
            with open("test_sentences", 'w') as f1:
                for test_text in all_test_list:
                    f1.write(test_text + '\n')

