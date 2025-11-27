import copy
import re
from typing import List, Optional
import numpy as np
from tsllm.envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP
from ...distributed.utils import print_with_rank

ANS_RE = re.compile(r'The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))')
STOP_STR = "The final answer is "
QUESTION_KEY = "question"


def extract_answer(completion):
    match = ANS_RE.findall(completion)
    group_select = -1
    if match:
        match = match[group_select]
        if isinstance(match, tuple):
            match = [m for m in match if m]
            if match:
                match = match[0]
            else:
                match = INVALID_ANS
        match = match.strip()
    else:
        match = INVALID_ANS

    return match


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


class Gsm8kEnv(CoTEnv):
    sep = SEP

    @staticmethod
    def build_query_str(
        cot_task_desc: Optional[str],
        cot_examples: Optional[str],
        problem_format_str: str,
        problem_input: str,
        sep: str,
        is_few_shot: bool = False,
    ):
        return problem_format_str.format(question=problem_input)

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
            action_distribution_temperature
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
            self.math_problem["question"], self.math_problem["answer"], extracted_answer
        )

    def init_action_history(self):
        question = self.math_problem['question']
        return [self.build_query_str(cot_task_desc=None, cot_examples=None, problem_format_str=self._problem_format_str,
                                    problem_input=question, sep=None)]

    def get_state(self):
        state = self.action_history[0]
        for action in self.action_history:
            assert action is not None, f'{self.action_history}'
            assert len(action) > 0, f'{self.action_history}'

        if len(self.action_history) > 1:
            state += self.sep.join(self.action_history[1:]) + self.sep

        return state

    def update_legal_actions(self):
        prefix = (
            (self.action_history[0] + "\n") if self.task_prefix is not None else None
        )
        act_hist_start_i = 0 if self.task_prefix is None else 1
        unprefixed_state = self.get_state()
        texts, logps, num_tokens, self_certainty_scores = self.llm_gen_fn(
            static_prompt=prefix,
            prompt=unprefixed_state,
            num_sequence=self.config["max_actions"],
            stop=[627, self.tokenizer.eos_token_id],
            add_special_tokens=False,
            return_self_certainty_scores=True,
            return_num_tokens=True,
            **self.config["generation_config"],
        )

        text_list = []
        valid_indices = []
        for i in range(len(texts)):
            if len(texts[i]) > 0 and texts[i] not in text_list:
                text_list.append(texts[i])
                valid_indices.append(i)

        if len(text_list) == 0:
            print_with_rank(
                "{} {} {}".format(prefix, act_hist_start_i, unprefixed_state)
            )
            raise NoLegalActionException("No possible action have been generated.")

        logps = np.array([logps[i] for i in valid_indices])
        num_tokens = np.array([num_tokens[i] for i in valid_indices])
        self_certainty_scores = np.array([self_certainty_scores[i] for i in valid_indices])
        # if self.config["generation_config"]["use_mean_logprob"]:
        #     logps /= num_tokens

        logps /= self.action_distribution_temperature
        probs = np.exp(logps - logps.max())
        probs /= probs.sum()

        _legal_actions = [
            {"action": action, "prob": float(prob), "num_token": int(n_token), "self_certainty_score": float(self_certainty_score)}
            for action, prob, n_token, self_certainty_score in zip(text_list, probs, num_tokens, self_certainty_scores)
        ]

        return _legal_actions

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0

    @property
    def question(self):
        return self.action_history[0]

    @property
    def answer(self):
        return self.sep.join(self.action_history[1:]) + self.sep
