# python eval.py --config ./config/default.yaml --save_path "./weights/Brownian Bridge Process - pred clean speech/epoch=120-pesq=2.57.ckpt"
python eval.py --config ./config/default.yaml --inference_config "./config/inference/two_stage.yaml" --model_path "./weights/epoch=72-pesq=2.75.ckpt" --N 3
# for d in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
# do
#     echo "$d"
#     python eval.py --config ./config/default.yaml --inference_config "./config/inference/two_stage.yaml" --model_path "./weights/Brownian Bridge Process - pred clean speech_fix_scale_by_sigma/epoch=72-pesq=2.75.ckpt" --N 3 --interpolate_weight $d --csv_path "./results/Brownian Bridge Process - pred clean speech_fix_scale_by_sigma/N=3/$d.csv"
# done