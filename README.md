This repository is based on: https://github.com/waterhorse1/LLM_Tree_Search.git

Installation:
1. Install torch==2.2.0
2. `pip install -r /path_to_repo/requirements.txt`
3. `cd /path_to_repo && pip install -e .`


# 1  GSM8k

## 1.1 SFT
```
cd /path_to_repo/train_mcts_scripts/gsm8k

# Default configuration for 8-GPU training: adjust in mcts_gsm8k_llama_deepspeed.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file mcts_gsm8k_llama_deepspeed.yaml train_gsm8k_sft.py --checkpoint_dir=sft
```
After 3 epochs of training, the sft folder contains checkpoint folders: checkpoint_0_ep0, checkpoint_1_ep1, and checkpoint_2_ep2.

## 1.2 Convert each SFT checkpoint using CTranslate2
```
ct2-transformers-converter --model sft/checkpoint_0_ep0 --quantization bfloat16 --output_dir sft_ctranslate2/llama2_sft_ep1_ct2
ct2-transformers-converter --model sft/checkpoint_1_ep1 --quantization bfloat16 --output_dir sft_ctranslate2/llama2_sft_ep2_ct2
ct2-transformers-converter --model sft/checkpoint_2_ep2 --quantization bfloat16 --output_dir sft_ctranslate2/llama2_sft_ep3_ct2
```

## 1.3 Generate data for training the value network
```
cd /path_to_repo/tsllm/offline_rl

# Data generation
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh gsm8k_data/gen_3.sh /path_to_repo/train_mcts_scripts/gsm8k/sft_ctranslate2 /path_to_repo/train_mcts_scripts/gsm8k/sft/checkpoint_0_ep0

# Data processing
sh gsm8k_data/process.sh
```

## 1.4 Train the value network
```cd /path_to_repo/train_mcts_scripts/gsm8k
accelerate launch --config_file mcts_gsm8k_llama_deepspeed.yaml train_gsm8k_critic.py --checkpoint_dir=value
```
After 3 epochs, the value folder contains checkpoint folders: checkpoint_0_ep0, checkpoint_1_ep1, and checkpoint_2_ep2.

## 1.5 Run MCTS on the test dataset split
```
export PYTHONPATH=/path_to_repo
export TEST_NO_TERMINAL=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /path_to_repo
save_dir=gsm8k_small
# rollout_method=mcts.gumbel for Gumbel AlphaZero, rollout_method=mcts.get_next_action for AlphaZero
torchrun --nproc_per_node=8 --master-port 29503 tsllm/offline_rl/test_sft_and_v.py \
    --ct2_dir /path_to_repo/train_mcts_scripts/gsm8k/sft_ctranslate2/llama2_sft_ep3_ct2\
    --critic_model_path /path_to_repo/train_mcts_scripts/gsm8k/value/checkpoint_2_ep2 \
    --tokenizer_path /path_to_repo/train_mcts_scripts/gsm8k/value/checkpoint_2_ep2 \
    --save_dir ${save_dir} \
    --env_name gsm8k \
    --rollout_method "mcts.gumbel" \
    --tree_max_length 8 \
    --tree_max_actions 6 \
    --test True \
    --final_action_strategy visits \
    --num_simulations 5 \
    --sequential_halving_start_nodes 3 \
    --non_root_child_selection_mode gumbel \
    --seed 0


save_dir=gsm8k_medium
# rollout_method=mcts.gumbel for Gumbel AlphaZero, rollout_method=mcts.get_next_action for AlphaZero
torchrun --nproc_per_node=8 --master-port 29503 tsllm/offline_rl/test_sft_and_v.py \
    --ct2_dir /path_to_repo/train_mcts_scripts/gsm8k/sft_ctranslate2/llama2_sft_ep3_ct2\
    --critic_model_path /path_to_repo/train_mcts_scripts/gsm8k/value/checkpoint_2_ep2 \
    --tokenizer_path /path_to_repo/train_mcts_scripts/gsm8k/value/checkpoint_2_ep2 \
    --save_dir ${save_dir} \
    --env_name gsm8k \
    --rollout_method "mcts.gumbel" \
    --tree_max_length 16 \
    --tree_max_actions 16 \
    --test True \
    --final_action_strategy visits \
    --num_simulations 30 \
    --sequential_halving_start_nodes 16 \
    --non_root_child_selection_mode gumbel \
    --seed 0


save_dir=gsm8k_large
# rollout_method=mcts.gumbel for Gumbel AlphaZero, rollout_method=mcts.get_next_action for AlphaZero
torchrun --nproc_per_node=8 --master-port 29503 tsllm/offline_rl/test_sft_and_v.py \
    --ct2_dir /path_to_repo/train_mcts_scripts/gsm8k/sft_ctranslate2/llama2_sft_ep3_ct2\
    --critic_model_path /path_to_repo/train_mcts_scripts/gsm8k/value/checkpoint_2_ep2 \
    --tokenizer_path /path_to_repo/train_mcts_scripts/gsm8k/value/checkpoint_2_ep2 \
    --save_dir ${save_dir} \
    --env_name gsm8k \
    --rollout_method "mcts.gumbel" \
    --tree_max_length 16 \
    --tree_max_actions 24 \
    --test True \
    --final_action_strategy visits \
    --num_simulations 50 \
    --sequential_halving_start_nodes 24 \
    --non_root_child_selection_mode gumbel \
    --seed 0
```



# 2 Game24

## 2.1 SFT
```
cd /path_to_repo/train_mcts_scripts/game24

# default config for 8 gpu training: adjust in mcts_gsm8k_llama_deepspeed.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file mcts_game24_llama_deepspeed.yaml train_game24_sft.py --checkpoint_dir=sft
```
After 3 epochs of training, the sft folder contains checkpoint folders: checkpoint_0_ep0, checkpoint_1_ep1, and checkpoint_2_ep2.

## 2.2 Convert each SFT checkpoint using CTranslate2
```
ct2-transformers-converter --model sft/checkpoint_0_ep0 --quantization bfloat16 --output_dir sft_ctranslate2/llama2_sft_ep1_ct2
ct2-transformers-converter --model sft/checkpoint_1_ep1 --quantization bfloat16 --output_dir sft_ctranslate2/llama2_sft_ep2_ct2
ct2-transformers-converter --model sft/checkpoint_2_ep2 --quantization bfloat16 --output_dir sft_ctranslate2/llama2_sft_ep3_ct2
```

## 2.3 Generate data for training the value network
```
cd /path_to_repo/tsllm/offline_rl

# Data generation
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sh game24/gen_3.sh /path_to_repo/train_mcts_scripts/game24/sft_ctranslate2 /path_to_repo/train_mcts_scripts/game24/sft/checkpoint_0_ep0

# Data processing
sh game24/process.sh
```

## 2.4 Train the value network
```
cd /path_to_repo/train_mcts_scripts/game24
accelerate launch --config_file mcts_game24_llama_deepspeed.yaml train_game24_critic.py --checkpoint_dir=value
```
After 3 epochs, the value folder contains checkpoint folders: checkpoint_0_ep0, checkpoint_1_ep1, and checkpoint_2_ep2.

## 2.5 Run MCTS on the test dataset split
```
export PYTHONPATH=/path_to_repo
export TEST_NO_TERMINAL=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /path_to_repo
save_dir=game24_small
# rollout_method=mcts.gumbel for Gumbel AlphaZero, rollout_method=mcts.get_next_action for AlphaZero
torchrun --nproc_per_node=8 --master-port 29503 tsllm/offline_rl/test_sft_and_v.py \
    --ct2_dir /path_to_repo/train_mcts_scripts/game24/sft_ctranslate2/llama2_sft_ep3_ct2\
    --critic_model_path /path_to_repo/train_mcts_scripts/game24/value/checkpoint_2_ep2 \
    --tokenizer_path /path_to_repo/train_mcts_scripts/game24/value/checkpoint_2_ep2 \
    --save_dir ${save_dir} \
    --env_name game24 \
    --rollout_method "mcts.gumbel" \
    --tree_max_length 4 \
    --tree_max_actions 20 \
    --test True \
    --final_action_strategy visits \
    --num_simulations 5 \
    --sequential_halving_start_nodes 3 \
    --non_root_child_selection_mode gumbel \
    --seed 0



save_dir=game24_medium
# rollout_method=mcts.gumbel for Gumbel AlphaZero, rollout_method=mcts.get_next_action for AlphaZero
torchrun --nproc_per_node=8 --master-port 29503 tsllm/offline_rl/test_sft_and_v.py \
    --ct2_dir /path_to_repo/train_mcts_scripts/game24/sft_ctranslate2/llama2_sft_ep3_ct2\
    --critic_model_path /path_to_repo/train_mcts_scripts/game24/value/checkpoint_2_ep2 \
    --tokenizer_path /path_to_repo/train_mcts_scripts/game24/value/checkpoint_2_ep2 \
    --save_dir ${save_dir} \
    --env_name game24 \
    --rollout_method "mcts.gumbel" \
    --tree_max_length 8 \
    --tree_max_actions 35 \
    --test True \
    --final_action_strategy visits \
    --num_simulations 30 \
    --sequential_halving_start_nodes 16 \
    --non_root_child_selection_mode gumbel \
    --seed 0



save_dir=game24_large
# rollout_method=mcts.gumbel for Gumbel AlphaZero, rollout_method=mcts.get_next_action for AlphaZero
torchrun --nproc_per_node=8 --master-port 29503 tsllm/offline_rl/test_sft_and_v.py \
    --ct2_dir /path_to_repo/train_mcts_scripts/game24/sft_ctranslate2/llama2_sft_ep3_ct2\
    --critic_model_path /path_to_repo/train_mcts_scripts/game24/value/checkpoint_2_ep2 \
    --tokenizer_path /path_to_repo/train_mcts_scripts/game24/value/checkpoint_2_ep2 \
    --save_dir ${save_dir} \
    --env_name game24 \
    --rollout_method "mcts.gumbel" \
    --tree_max_length 16 \
    --tree_max_actions 50 \
    --test True \
    --final_action_strategy visits \
    --num_simulations 50 \
    --sequential_halving_start_nodes 24 \
    --non_root_child_selection_mode gumbel \
    --seed 0
```










