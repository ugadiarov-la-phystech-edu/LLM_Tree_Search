import copy
import re
from typing import List, Optional
import numpy as np
from vllm import SamplingParams

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

    @staticmethod
    def build_response_str(
        answer_str: str, tokenizer, add_eos_token: bool
    ):
        raise NotImplementedError

    def __init__(
        self,
        config,
        math_problems,
        llm,
        tokenizer,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
        action_distribution_temperature=1.0,
        reasoning_effort='medium',
        llm_gen_fn=None,
    ):
        token_ids = tokenizer.encode(self.sep)
        assert len(token_ids) == 2, f'len(token_ids): {len(token_ids)}'
        self.llm = llm
        stop_token_ids = [token_ids[-1], tokenizer.eos_token_id]
        generation_config = self.config['generation_config']
        self.sampling_params = SamplingParams(
            max_tokens=generation_config['max_new_tokens'],
            temperature=generation_config['temperature'],
            stop_token_ids=stop_token_ids,
            top_k=generation_config['top_k'],
            top_p=generation_config['top_p'],
            n=self.config["max_actions"],
            logprobs=1,
        )
        self._stop_token_ids = set(stop_token_ids)

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
        assert self.task_prefix is None, f'Task prefix:{self.task_prefix}'

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
        # add the first prompted questions
        question = self.math_problem['question']
        return [self.build_query_str(cot_task_desc=None, cot_examples=None, problem_format_str=self._problem_format_str,
                                    problem_input=question, sep=None)]

    def get_state(self):
        state = self.action_history[0]
        if len(self.action_history) > 1:
            state += self.sep.join(self.action_history[1:]) + self.sep

        return state

    def _generate(self, prompt):
        outputs = self.llm.generate([prompt], sampling_params=self.sampling_params, use_tqdm=False,)[0].outputs
        texts = []
        logprobs = []
        for completion_output in outputs:
            token_ids = completion_output.token_ids
            # return completions without stop tokens
            stop_token_logprob = 0
            if token_ids[-1] in self._stop_token_ids:
                token_ids = token_ids[:-1]
                stop_token_logprob = list(completion_output.logprobs[-1].values())[0].logprob

            text = self.tokenizer.batch_decode([token_ids], skip_special_tokens=False)[0]
            logprob = completion_output.cumulative_logprob - stop_token_logprob
            texts.append(text)
            logprobs.append(logprob)

        return texts, logprobs

    def update_legal_actions(self):
        prefix = None
        unprefixed_state = self.get_state()
        texts, logps = self._generate(unprefixed_state)
        text_list, logprob_list = [], []

        for i in range(len(texts)):
            if len(texts[i]) > 0 and texts[i] not in text_list:
                text_list.append(texts[i])
                logprob_list.append(logps[i])

        if len(logprob_list) == 0:
            print_with_rank(
                "{} {} {}".format(prefix, 0, unprefixed_state)
            )
            raise NoLegalActionException("No possible action have been generated.")

        logprobs = np.array(logprob_list, dtype=np.float64)
        unnormalized_probs = np.exp((logprobs - logprobs.max()) / self.action_distribution_temperature)
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        if np.isnan(probs).any().item():
            log = f'\nlogprobs: {logprobs}'
            log += f'\nunnormalized prob_list: {unnormalized_probs}'
            log += f'\nprobs: {probs}'
            raise ValueError(log)

        # set add special tokens as False to remove bos/eos tokens
        num_token_list = [
            len(self.tokenizer.encode(txt, add_special_tokens=False))
            for txt in text_list
        ]
        _legal_actions = [
            {"action": action, "prob": prob, "num_token": n_token}
            for action, prob, n_token in zip(text_list, probs, num_token_list)
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
