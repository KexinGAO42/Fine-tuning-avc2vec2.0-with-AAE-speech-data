import json
import flash
from flash.audio import SpeechRecognition, SpeechRecognitionData
from pytorch_lightning import seed_everything
import torch

seed_everything(950601)

'''1. Load test data and create data module'''
test_json_path = "../test_case/test_case.json"
datamodule = SpeechRecognitionData.from_json(
    "file",  # json field for wav file
    "text",  # json field for transcript
    predict_file=test_json_path,
    sampling_rate=16000,
    batch_size=2,
)

'''2. Load the original model and finetuned model'''

original_model = SpeechRecognition.load_from_checkpoint("../models/original_model.pt")
finetuned_model = SpeechRecognition.load_from_checkpoint("../models/finetuned_model.pt")

'''3. Create the trainer and predict'''
trainer1 = flash.Trainer(gpus=torch.cuda.device_count())
original_pred = trainer1.predict(original_model, datamodule=datamodule)
trainer2 = flash.Trainer(gpus=torch.cuda.device_count())
finetuned_pred = trainer2.predict(finetuned_model, datamodule=datamodule)

result_wo_finetune = [i for sublist in original_pred for i in sublist]
result_wo_finetune = [s.lower() for s in result_wo_finetune]
result_wi_finetune = [i for sublist in finetuned_pred for i in sublist]
result_wi_finetune = [s.lower() for s in result_wi_finetune]

gold_standard = []
with open(test_json_path, 'r') as test_json:
    line = test_json.readline()
    while line:
        tmp = json.loads(line)
        gold_standard.append(tmp["text"].lower())
        line = test_json.readline()

num_test_case = len(gold_standard)
for i in range(num_test_case):
    print("test case {} \n gold standard: {}\n original result: {}\n fine-tuned result: {}\n"
          .format(i+1, gold_standard[i], result_wo_finetune[i], result_wi_finetune[i]))
