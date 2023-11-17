#\bin\sh

lr_list="5e-5 3e-5 1e-5"
for lr in $lr_list;
do
  for ((epoch=0;epoch<=22;epoch=epoch+2));
  do
    python finetune.py --num_epoch "$epoch" --lr "$lr" > "result\\${epoch}_${lr}.txt"
  done

done
sleep 10