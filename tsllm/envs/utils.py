import math
from typing import Dict, Optional, Union, Callable
import numpy as np
from tsllm.distributed.utils import print_with_rank
from tsllm.offline_rl.utils import load_jsonl
from transformers import PreTrainedTokenizer
from pathlib import Path
from torch.utils.data import Dataset
import jsonlines
from tqdm import tqdm


def build_sft_data_component(
    jsonl_path: Union[Path, str],
    q2idx_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    add_eos_token: bool,
    is_few_shot: bool,
    build_query_str_fn: Callable,
    build_response_str_fn: Callable,
    sep: str,
    cot_task_desc_str: Optional[str] = None,
    cot_example_str: Optional[str] = None,
    problem_format_str: Optional[str] = None,
):
    predata = load_jsonl(jsonl_path)
    q_r_dict_list = []
    for idx, d in enumerate(predata):
        question = d["question"]
        if question not in q2idx_dict:
            continue
        task_idx = q2idx_dict[question]
        full_answer_list = d["answer"]
        query_str = build_query_str_fn(
            cot_task_desc=cot_task_desc_str,
            cot_examples=cot_example_str,
            problem_format_str=problem_format_str,
            problem_input=question,
            sep=sep,
            is_few_shot=is_few_shot,
        )

        for answer_output in full_answer_list:
            answer_txt = answer_output["text"]
            response_str = build_response_str_fn(answer_txt, tokenizer, add_eos_token)
            traj_dict = {
                "idx": task_idx,
                "query_str": query_str,
                "answer": answer_txt,
                "response_str": response_str,
            }
            q_r_dict_list.append(traj_dict)

    return q_r_dict_list


def build_critic_data_component_in_tokens(
    jsonl_path: Union[Path, str],
    q2idx_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    sep_token: str,
    last_query_token: str,
    is_few_shot: bool,
    build_query_str_fn: Callable,
    cot_task_desc_str: Optional[str] = None,
    cot_example_str: Optional[str] = None,
    problem_format_str: Optional[str] = None,
):
    sep_token_id = tokenizer.encode(sep_token)[0]
    last_query_token_id = tokenizer.encode(last_query_token)[0]
    n_total_answers = 0
    n_skipped_answers = 0
    min_skipped_length = math.inf

    def get_value_index(q_str: str, answer_str: str):
        query_tokens = tokenizer.encode(q_str,)
        answer_tokens = tokenizer.encode(answer_str)
        tokens = query_tokens + answer_tokens
        assert len(tokens) == len(tokenizer.encode(q_str + answer_str))

        has_last_query_token = False
        has_sep_token = False
        has_eos_token = False
        indices = []
        for index, token in enumerate(tokens):
            if not has_last_query_token:
                if token == last_query_token_id:
                    has_last_query_token = True
                    indices.append(index)
            else:
                if token == sep_token_id:
                    has_sep_token = True
                    indices.append(index)
                elif token == tokenizer.eos_token_id:
                    has_eos_token = True
                    assert index == len(tokens) - 1
                    indices.append(index)

        assert has_last_query_token, f'Query={q_str}\n@@@\nAnswer={answer_str}\n&&&'
        if indices[-1] < len(tokens) - 1:
            assert not has_eos_token
            indices.append(len(tokens) - 1)

        return indices

    predata = load_jsonl(jsonl_path)
    traj_dict_list = []
    pbar = tqdm(predata)
    for idx, d in enumerate(pbar):
        question = d["question"]
        if question not in q2idx_dict.keys():
            continue
        task_idx = q2idx_dict[question]
        full_answer_list = d["answer"]
        query_str = build_query_str_fn(
            cot_task_desc=cot_task_desc_str,
            cot_examples=cot_example_str,
            problem_format_str=problem_format_str,
            problem_input=question,
            sep=None,
            is_few_shot=is_few_shot,
        )
        for answer_output in full_answer_list:
            """answer_output is a dict with keys:
            "text", "reward",
            if there is not "reward" key, use "correct" key
            """
            n_total_answers += 1
            answer = answer_output["text"]
            value_index = get_value_index(query_str, answer)
            if len(value_index) < 2:
                # print_with_rank(f'Query-Answer string does not contain both "sep" or "end_of_answer" tokens, so we are skipping it: Query={query_str}\n@@@\nAnswer={answer}\n&&&')
                n_skipped_answers += 1
                min_skipped_length = min(min_skipped_length, len(tokenizer.encode(answer)))
                continue

            # :-1 is value index, -1 is the reward index
            reward_list = np.zeros(len(value_index) - 1)
            if "reward" not in answer_output:
                answer_output["reward"] = 1.0 if answer_output["correct"] else -1.0

            reward_list[-1] = answer_output["reward"]
            traj_dict = {
                "idx": task_idx,
                "query_str": query_str,
                "answer": answer,
                "value_index": value_index,
                "reward_list": reward_list,
            }
            traj_dict_list.append(traj_dict)

        pbar.set_postfix({'Total answers': n_total_answers, 'Skipped answers': n_skipped_answers, 'Skip rate': n_skipped_answers / n_total_answers, 'Min length of skipped answers': min_skipped_length})

    print_with_rank(f'Total answers: {n_total_answers}. Skipped answers: {n_skipped_answers}. Min length of skipped answer: {min_skipped_length}.')
    return traj_dict_list
