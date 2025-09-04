import copy
import re
from typing import List, Optional
import numpy as np
from tsllm.envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP

WHITESPACE = re.compile(r"\s+")
STOP_STR = "The answer is "
QUESTION_KEY = "problem"


def extract_answer(completion):
    substring = completion.split(STOP_STR)
    if len(substring) < 2:
        return INVALID_ANS

    answer = WHITESPACE.sub("", substring[1])
    return answer


def extract_groundtruth(groundtruth_str: str):
    return WHITESPACE.sub("", groundtruth_str)


def judge_correct(problem_str: str, extracted_groundtruth: Optional[str], answer: str):
    return answer == extracted_groundtruth


class Math500(CoTEnv):
    sep = SEP

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        tokenizer,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
        action_distribution_temperature=1.0,
    ):
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            tokenizer,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
            action_distribution_temperature,
        )

    @property
    def stop_str(self):
        return STOP_STR

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        # print("Compare: {} -- {}".format(extrated_answer,
        #  self.math_problem['answer']))
        # return extrated_answer == self.math_problem['answer']
        return judge_correct(
            self.math_problem['question'], self.math_problem["answer"], extracted_answer
        )

    def init_action_history(self):
        # add the first prompted questions
        return ([self.task_prefix] if self.task_prefix is not None else []) + [
            f"Question: {self.math_problem['question']}\nAnswer: Let's think step by step"
        ]

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0
