import os
from evaluate import load
import json

wer = load("wer")

K = 28
# SELECTED_EPOCH=20
# SELECTED_LR=5e-5
SELECTED_EPOCH = None
SELECTED_LR = None

if __name__ == "__main__":

    test_json_path = "data\\CORAAL\\For_Dev\\Split\\test.json"
    result_folder = "result"
    output_path = "full_result_CORAAL_V2.txt"

    test_sentences = []
    with open(test_json_path, 'r') as test_json:
        line = test_json.readline()
        while line:
            tmp = json.loads(line)
            test_sentences.append(tmp["text"].lower())
            line = test_json.readline()

    with open(output_path, 'w') as of:

        pretrain_predict_sentences = []  # Save what the pretrained does
        best_WER = 1.0  # To report best WER
        best_epoch = None
        best_lr = None

        with os.scandir(result_folder) as all_results:
            for one_result in all_results:  # Each result file

                file_name = os.path.join(result_folder, one_result.name)

                tmp_list = one_result.name.split('_')
                num_epoch = int(tmp_list[0])
                lr = float(tmp_list[1].split('.')[0])

                last_line = None
                with open(file_name, 'r') as res_f:
                    for line in res_f:
                        pass
                    last_line = line

                # remove all the brackets in str, then each sentences can be tokenized
                last_line = last_line.replace('[', '')
                last_line = last_line.replace(']', '')
                pred_sents = last_line.split(", ")
                pred_sents = [s.lower()[1:-1] for s in pred_sents]  # remove quotes

                wer_score = wer.compute(predictions=pred_sents, references=test_sentences)

                if num_epoch == 0:
                    pretrain_predict_sentences = pred_sents

                if best_WER > wer_score:
                    best_WER = wer_score
                    best_epoch = num_epoch
                    best_lr = lr

                ##########
                if SELECTED_EPOCH is not None and num_epoch != SELECTED_EPOCH:
                    continue
                if SELECTED_LR is not None and lr != SELECTED_LR:
                    continue

                one_result_line = "Epoch {}\tLR {}\tWER {} \n".format(num_epoch, lr, wer_score)
                of.write(one_result_line)

                # Also report predicted sentences

                of.write("Top {} label vs prediction: \n".format(K))
                for i in range(K):
                    of.write("Label:\t\t {} \n".format(
                        test_sentences[i][1:]))  # We accidentally add a space in test sentences :(
                    of.write("Pret pred:\t {} \n".format(pretrain_predict_sentences[i]))
                    of.write("Pred:\t\t {} \n".format(pred_sents[i]))
                of.write("\n")

        of.write("Best WER {}, best epoch {}, best lr {}.\n".format(best_WER, best_epoch, best_lr))