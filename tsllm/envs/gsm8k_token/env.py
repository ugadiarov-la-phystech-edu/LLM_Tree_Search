import copy
import re
from typing import List, Optional
import numpy as np
from tsllm.envs.base_env import NoLegalActionException, INVALID_ANS, TokenEnv
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP


ANS_RE = re.compile(r"The answer is ((\-?[0-9\.\,]+)(\s+|$))?")


def extract(text):
    number_group = 2
    result = None
    span = None
    terminated = False
    match = re.search(ANS_RE, text)
    if match is not None:
        result = match.group(number_group)
        span = match.span(number_group)

    if result is not None and span[1] < len(text):
        # We have a number after 'The answer is ' and a space after the number, so we know that the model stop generated the number.
        # Extract answer and terminate generation.
        terminated = True
    elif match is not None and result is None and match.span(0)[1] < len(text):
        # We have 'The answer is ', but do not have a number after it; and the text does not end with 'The answer is'.
        # Stop generation as we know for sure that the answer will be wrong.
        terminated = True

    return terminated, result


def extract_answer(completion):
    _, match_str = extract(completion)
    if match_str is not None:
        match_str = match_str.strip()
        match_str = match_str.replace(",", "")
    else:
        return INVALID_ANS

    return match_str


def extract_groundtruth(groundtruth_str: str):
    x = groundtruth_str.split("#### ")[1].strip().replace(",", "")
    try:
        float(x)
    except:
        raise ValueError(
            "Warning: Error should raise since the extracted groundtruth string {}\
             cannot be converted to float".format(
                x
            )
        )
    return x


def judge_correct(problem_str: str, extracted_groundtruth: Optional[str], answer: str):
    float_groundtruth = float(extracted_groundtruth)
    try:
        return abs(float(answer) - float_groundtruth) < 1e-5
    except Exception:
        return False


class Gsm8kTokenEnv(TokenEnv):
    sep = SEP

    def __init__(
        self,
        config,
        problems,
        llm_forward_fn,
        tokenizer,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
    ):
        super().__init__(
            config,
            problems,
            llm_forward_fn,
            tokenizer,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
        )

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        # print("Compare: {} -- {}".format(extrated_answer,
        #  self.math_problem['answer']))
        # return extrated_answer == self.math_problem['answer']
        return judge_correct(
            self.problem["question"], self.problem["answer"], extracted_answer
        )

    def init_action_history(self):
        # add the first prompted questions
        return ([self.task_prefix] if self.task_prefix is not None else []) + [
            f"Question: {self.problem['question']}\nAnswer: Let's think step by step"
        ]

    def get_reward(self, *args, **kwargs):
        """To implement based on learned reward model"""
        return 0

    @property
    def sep_index(self):
        raise NotImplementedError

    def step(self, action, update_legal_action=True):
        terminated = False
        if action != self.tokenizer.eos_token:
            # remove the final stop string like eos token
            self.action_history.append(action)
        else:
            terminated = True

        state = self.get_state()
        terminated_from_state, answer = extract(state)
        terminated = terminated or terminated_from_state
        truncated = len(self.action_history) >= self.config["max_length"] + (
            2 if self.task_prefix is not None else 1
        )
        reward = self.get_reward(terminated, truncated)

        info = {'reward': reward, 'winner': 0}
        if terminated or truncated:
            if self._is_correct(state):
                info["winner"] = 1
            else:
                info["winner"] = 2

        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            self._legal_actions = self.update_legal_actions()
        else:
            self._legal_actions = None
            if info["winner"] == 1:
                reward = 1.0

        return state, reward, terminated, truncated, info

    @property
    def question(self):
        raise NotImplementedError
