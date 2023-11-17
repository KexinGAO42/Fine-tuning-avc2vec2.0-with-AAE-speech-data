# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

import flash
from flash.audio import SpeechRecognition, SpeechRecognitionData
from pytorch_lightning import seed_everything
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, required=False, default=20)
parser.add_argument('--lr', type=float, required=False, default=5e-5)
args = parser.parse_args()

seed_everything(950601)

#################################### 1. Load data and create data module

#1. Create the DataModule
datamodule = SpeechRecognitionData.from_json(
    "file",  # json field for wav file
    "text",  # json field for transcript
    train_file="../processed_data/Split/train.json",   # train json data
    val_file="../processed_data/Split/val.json",
    predict_file="../processed_data/Split/test.json",
    sampling_rate=16000,
    batch_size=2,
)

#################################### 2. Build the model

# 2. Build the tas
# k
model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h", learning_rate=args.lr)

#################################### 3. Build the model
# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=args.num_epoch, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy=("freeze_unfreeze", 7))

# 4. Predict on audio files!
#datamodule = SpeechRecognitionData.from_files(predict_files=["data/timit/example.wav"], batch_size=4)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("saved_models/{}_{}.pt".format(args.num_epoch, args.lr))
