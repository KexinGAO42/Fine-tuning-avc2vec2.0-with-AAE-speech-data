# Readme


---


---




## Introduction

This project is the required final work for LING580A (Bias in ASR), which aims at improving the accuracy of ASR models at recognizing sociolinguistic variations (e.g., consonant cluster deletion).

Our approach is to finetune wav2vec2.0-based model with relatively small amount of within-speaker African American English(AAE) speech data. 


---



## Files
### raw_data/
1. ROC_se0_ag1_f_02_1.wav - [speech data from CORAAL ](http://lingtools.uoregon.edu/coraal/explorer/browse.php?what=ROC_se0_ag1_f_02_1.txt)
2. ROC_se0_ag1_f_02_1.TextGrid - correspondant transcription

### processed_data/
1. Split/  \\
ROC_se0_ag1_f_02/ - segmented files (.wav - .txt pairs) \\
test.json - test instance \\
train.json - training instance \\
val.json - validation instance \\
2. ROC_se0_ag1_f_02_1_denoise.wav - denoised audio file

### models/
1. original_moddel.pt - pretrained model without finetuning
2. finetuned_model.pt - finetuned model

### logs/
results from different parameters
### scripts/
*Details introduced in next part.*
1. `data_processing.py ` \\
2. `finetune.py` \\
3. `find_best_params.py ` \\
4. `run.sh` \\
4. `test.py ` \\

### requirements.txt
required package for running test cases.

### test_case/
1. test_case.json - the json files that maps 6 selected test audios and the text 
2. ROC_se0_ag1_f_02/ - correspondant .wav file for the test samples


---



## Project Outline
### 1. Data processing
Correspondent script: `data_processing.py`  \\
We selected the speech data from one single AAE speaker from the CORAAL dataset. The data if conversatioinal speech. The data consists of one .wav audio file and one .textgrid file transcripted in utterance level. \\
For the pre-processing: \\
1. We segmented the audio data based on the .textgrid file, and excluded the audio from the interviewer. \\
2. We performed noise reduction on the audio file. We normalized the transcripition by removing all punctuations, skipped audio with transcirbed laughing and exhaling. We also controlled the length of audio to be at least 10 seconds to ensure complete context. \\
3. We then splitted the audio files into three parts: 80% for training, 5% for validation, and 20% for test. We also formatted the audio files and transcriptions into .json, which is preferred by the fine-tuning package that we will use later. \\

By running data_processing.py script, you input the original audio file and its correspondant textgrid file (i.e. in `raw_data/ `) selected from CORAAL, you get the data in `processed_data/ `:
  1. a folder named after the original name of the data point, containing the segmented audio files \\
  2. three json files mapping the audio to the text for training, validation and test respectively \\

### 2. Fine-tuning and get predictions
Correspondent script: `finetune.py` \\
To adapt an ASR system to AAE accent, we performed small within-speaker fine-tuning on pretrained model. The backbone model we chose is Facebook wav2vec2.0 pretrained on 960 hours speech data from Licri Speech dataset. This dataset is designed for a balenced gender. And the content consist of audiobook reading. \\
For fine-tuning:
1. Load the dataset created from the previous step \\
2. We selected Facebook wav2vec pretrained on 960 hours Speech dataset as the backbone model. \\
3. Create the trainer and setup finetuning parameters. \\
4. Return the transcription for the audios in test data. \\
5. Save the fine-tuned model. \\


### 3. Run multiple experiments to find the best parameters
Correspondent script: `finetune.py` `find_best_params.py` `run.sh` \\
To find the best parameters of # epoch and learning rate, we run different combination of these two parameters by replicating the previous step. \\
We get the results and calculate the WER, and find the parameters that leads to the lowest WER. \\
1. Run `run.sh` script, which calls `finetune.py` 3 times, with 3 different learning rate of 5e-5, 3e-5, 1e-5; for each learning rate, the experiment runs up to 22 epochs, and save the model and predictions every two epochs. \\
2. Run `find_best_params.py`, which takes the results of previous experiments, it calculates the WER of each experiment and find the best one.


### 4. Test on a subset of audios to obverve the improvement in recognizing language variation
Correspondent script: `test.py` \\
In this part, we selected a subset of audios which contains several linguistic variations, and compare the automatic transcription results from the two models before and after finetuning. \\


---



## Instruction on testing out the model
*NB: The data-processing, finetuning, and experimenting part is fixed in the script and not subscribed to change unless you manually modify the script. We only replicate Part 4.* \\

Please make sure that you don't change the relative path for the files.
Please use Linux system to run the files.

1. get the required packages: in this part, we used [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html) and [Lighting Flash](https://lightning-flash.readthedocs.io/en/stable/quickstart.html) as deep learning framework. The required packages are listed in `requiremeents.txt` Run the following code to install:

```
pip install -r requirements.txt
```
2. Open and run the  `test.py` file, you will get the output of several test case printed.


---



## Result discussion

By observing the output of the test case, we can see that even though the finetuned model still makes a fair amount of mistakes, it shows some improvement in the recognition errors compared to the backbone. \\

The backbone model is bad at detecting AAE speech for two possible reason. One is that it is pretrained on audiobook speech data, whereas our test data is conversational speech. Another reason is that AAE speech data is very likely to be highly under-presented by the training data for the pretrained model. \\

By finetuning the model with within-speaker AAE speech data, we can see that the transcription of one language variation has been siginificantly improved, which is the confusion betweem (æ) and (ɪ) (e.g. target - in, example error - and). One possible reason for this improvement could be that it occurs relatively more frecuent than other types of phone and could be easier to learn through the iteration. \\

However, we still see other types of sociolinguistic variables produced by the fine-tuned model. For further improvement, larger amount of training data could be used to fine-tune the model. And also, using non-conversational speech data for fine-tuning might also improve the transcription accuracy.

