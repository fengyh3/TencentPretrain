deepspeed run_rlhf.py --deepspeed --enable_zero3 \
                      --vocab_path ../models/google_zh_vocab.txt \
                      --world_size 1 \
                      --deepspeed_checkpoint_activations \
                      --total_steps 100 \
                      --save_checkpoint_steps 100 \
                      --batch_size 32 \
                      --ppo_dataset_path \
                      --actor_pretrained_model_path /home/ubuntu/workspace_fyh/gpt2-chinese-cluecorpussmall/pytorch_model.bin \
                      --critic_pretrained_model_path /home/ubuntu/workspace_fyh/gpt2-chinese-cluecorpussmall/pytorch_model.bin \
                      --actor_output_model_path ./actor_output.bin \
                      --critic_output_model_path ./critic_output.bin \
                      --actor_config_path /home/ubuntu/workspace_fyh/gpt2-chinese-cluecorpussmall/config.json \
                      --critic_config_path /home/ubuntu/workspace_fyh/gpt2-chinese-cluecorpussmall/config.json \
                      --actor_target ppo \
                      --critic_target rm \
                      --ppo_data_processor ppo \
                      --actor_learning_rate 1e-5 \
                      --critic_learning_rate 1e-5 \
                      --actor_deepspeed_config ./deepspeed_config/train_deepspeed_zero3_config.json \
                      --critic_deepspeed_config ./deepspeed_config/train_deepspeed_zero3_config.json \
                      --ref_deepspeed_config ./deepspeed_config/eval_deepspeed_zero3_config.json \
                      --reward_deepspeed_config ./deepspeed_config/eval_deepspeed_zero3_config.json \


# --unsupervised_data_processor alpaca \
# --unsupervised_dataset_path \
# --spm_model_path $LLaMA_PATH/tokenizer.model \