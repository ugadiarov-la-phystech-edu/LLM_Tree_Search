import copy
import re
from typing import List, Optional
import numpy as np
from tsllm.envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP
from ...distributed.utils import print_with_rank

ANS_RE = re.compile(r"The final answer is (\-?[0-9\.\,]+)")
STOP_STR = "The final answer is "
QUESTION_KEY = "question"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
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
        if len(self.action_history) > 1:
            state += self.sep.join(self.action_history[1:]) + self.sep

        return state

    def update_legal_actions(self):
        def reduce_prob_list(prob_list: List[List]) -> List:
            ans_list = []
            for scores in prob_list:
                ans_list.append(np.exp(np.mean(scores)) / self.action_distribution_temperature)
            return ans_list

        prefix = (
            (self.action_history[0] + "\n") if self.task_prefix is not None else None
        )
        act_hist_start_i = 0 if self.task_prefix is None else 1
        unprefixed_state = self.get_state()
        texts, logps, num_tokens = self.llm_gen_fn(
            static_prompt=prefix,
            prompt=unprefixed_state,
            num_sequence=self.config["max_actions"],
            stop=[627, self.tokenizer.eos_token_id],
            add_special_tokens=False,
            retrun_num_tokens=True,
            **self.config["generation_config"],
        )

        text_list, prob_list = [], []
        for i in range(len(texts)):
            if len(texts[i]) > 0 and texts[i] not in text_list:
                text_list.append(texts[i])
                log_prob = logps[i]
                if self.config["generation_config"]["use_mean_logprob"]:
                    log_prob /= num_tokens[i]

                prob_list.append(log_prob)

        if len(prob_list) == 0:
            print_with_rank(
                "{} {} {}".format(prefix, act_hist_start_i, unprefixed_state)
            )
            raise NoLegalActionException("No possible action have been generated.")

        prob_list = reduce_prob_list(prob_list)
        prob_list = np.array(prob_list)
        # normalize probability
        prob_list = prob_list / np.sum(prob_list)
        # set add special tokens as False to remove bos/eos tokens
        num_token_list = [
            len(self.tokenizer.encode(txt, add_special_tokens=False))
            for txt in text_list
        ]
        _legal_actions = [
            {"action": action, "prob": prob, "num_token": n_token}
            for action, prob, n_token in zip(text_list, prob_list, num_token_list)
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
