import abc
import re
from typing import Dict, List, Optional
import numpy as np
import copy
import pdb
import torch
from tsllm.distributed.utils import print_with_rank
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams


INVALID_ANS = "[invalid]"
SEP = "\n\n"
STOP_STR = "Answer:"
# STOP_STR = "The answer is "
QUESTION_KEY = "question"
ANS_RE = re.compile(r"([-+]?\d*\.\d+|\d+)")


class NoLegalActionException(Exception):
    pass


class ResetException(Exception):
    pass


class BaseEnv(abc.ABC):
    """Basic environment to use for MCTS"""

    @abc.abstractmethod
    def reset(self, update_legal_action: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError

    @abc.abstractproperty
    def legal_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @staticmethod
    def build_query_str(
            cot_task_desc: Optional[str],
            cot_examples: Optional[str],
            problem_format_str: str,
            problem_input: str,
            sep: str,
            is_few_shot: bool = False,
    ):
        raise NotImplementedError

    @staticmethod
    def build_response_str(
            answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool
    ):
        raise NotImplementedError


class CoTEnv(BaseEnv):
    """The basic environment for solving natural language problems using CoT"""

    sep: str

    @staticmethod
    def build_response_str(
            answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool
    ):
        raise NotImplementedError

    @property
    def stop_str(self):
        return NotImplementedError

    def _is_correct(self, completion) -> bool:
        raise NotImplementedError

    def get_reward(self):
        """To implement based on learned reward model"""
        raise NotImplementedError

    def __init__(
            self,
            config,
            math_problems,
            llm: LLM,
            tokenizer,
            task_desc_str: str,
            cot_example_str: str,
            problem_format_str: str,
            reset=True,
            action_distribution_temperature=1.0,
            reasoning_effort: str ='medium',
    ):
        self.config = config
        self.mcts_mode = "play_with_bot_mode"
        self.math_problems = math_problems
        self.llm = llm
        self.tokenizer = tokenizer
        self.action_history = None
        self.math_problem = None
        self._legal_actions = None
        self.is_few_shot = config.get("is_few_shot", False)
        self.action_distribution_temperature = action_distribution_temperature
        self.reasoning_effort = reasoning_effort

        self._task_desc_str = task_desc_str
        self._cot_example_str = cot_example_str
        self._problem_format_str = problem_format_str

        assert not self.is_few_shot
        assert self._task_desc_str is None
        assert self._cot_example_str is None
        assert self._problem_format_str is None

        stop_token_ids = [self.tokenizer.eos_token_id]
        # stop_token_ids.extend(self.tokenizer.encode('\n'))
        stop_token_ids.extend(self.tokenizer.encode('\n\n'))

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

        if reset:
            self.reset(update_legal_action=True)

    def reset(self, update_legal_action=True):
        # reset environment to problem idx
        self.set_problem(idx=0)
        self.action_history = self.init_action_history()
        if update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions = self.update_legal_actions()
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                        raise ResetException
        return self.get_state()

    def step(self, action, update_legal_action=True):
        self.action_history.append(action)
        state = self.get_state()
        reward = self.get_reward()
        terminated, truncated, info = self.get_done_and_info()
        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            try:
                self._legal_actions = self.update_legal_actions()
            except NoLegalActionException as e:
                terminated = True
                reward = 0
                self._legal_actions = None
                info["winner"] = 2
        else:
            self._legal_actions = None
            if info["winner"] == 1:
                reward = 1.0
        return state, reward, terminated, truncated, info

    # def get_state(self):
    #     return self.action_history[0] + "\n" + "\n".join(self.action_history[1:]) + "\n"

    def get_state(self):
        if len(self.action_history) == 1:
            return self.action_history[0]

        return self.action_history[0] + SEP.join(self.action_history[1:]) + SEP

    # def init_action_history(self):
    #     # add the first prompted questions
    #     return [
    #         f"Question: {self.math_problem['question']}\nAnswer: Let's think step by step"
    #     ]

    def init_action_history(self):
        # add the first prompted questions
        question = self.math_problem["question"]
        message = {"role": "user", "content": f'{question}. Provide numeric answer after "Answer:"'}
        action = self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True, reasoning_effort=self.reasoning_effort)
        return [action]

    def _generate(self, prompt):
        outputs = self.llm.generate([prompt], sampling_params=self.sampling_params, use_tqdm=False,)[0].outputs
        texts = []
        logprobs = []
        for completion_output in outputs:
            text = self.tokenizer.batch_decode([completion_output.token_ids], skip_special_tokens=False)[0]
            logprob = completion_output.cumulative_logprob
            texts.append(text)
            logprobs.append(logprob)

        return texts, logprobs

    def update_legal_actions(self):
        def reduce_prob_list(prob_list: List[List]) -> List:
            ans_list = []
            for scores in prob_list:
                ans_list.append(np.exp(np.mean(scores)) / self.action_distribution_temperature)
            return ans_list

        prefix = None
        unprefixed_state = self.get_state()
        texts, logps = self._generate(unprefixed_state)
        text_list, prob_list = [], []

        for i in range(len(texts)):
            if len(texts[i]) > 0 and texts[i] not in text_list:
                text_list.append(texts[i])
                prob_list.append(logps[i])

        if len(prob_list) == 0:
            print_with_rank(
                "{} {} {}".format(prefix, 0, unprefixed_state)
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

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    @property
    def question(self):
        return self.action_history[0]

    @property
    def answer(self):
        return SEP.join(self.action_history[1:]) + SEP

    def get_done_and_info(self):
        info = {"winner": 0}
        # done when reaches maximum length or LLM generates stop words
        terminated = self.stop_str in self.action_history[-1]
        max_length = self.config["max_length"] + 1
        truncated = len(self.action_history) >= max_length
        assert len(self.action_history) <= max_length, f"action history length: {len(self.action_history)}, max length: {max_length}"
        if terminated or truncated:
            if self._is_correct(self.action_history[-1]):
                info["winner"] = 1
            else:
                info["winner"] = 2
            return terminated, truncated, info
        return terminated, truncated, info

    def copy(self):
        env = self.__class__(
            self.config,
            self.math_problems,
            self.llm,
            self.tokenizer,
            self._task_desc_str,
            self._cot_example_str,
            self._problem_format_str,
            reset=False,
        )
        env.math_problem = copy.deepcopy(self.math_problem)
        env._legal_actions = copy.deepcopy(self._legal_actions)
        env.action_history = copy.deepcopy(self.action_history)
        return env

    @property
    def legal_actions(self):
        return self._legal_actions


# def extract_answer(completion):
#     ANS_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)")
#     match = ANS_RE.search(completion)
#     if match:
#         match_str = match.group(1).strip()
#         match_str = match_str.replace(",", "")
#     else:
#         return INVALID_ANS
#     return match_str


def extract_answer(completion):
    tag = "<|channel|>final<|message|>"
    answer_index = completion.rfind(tag) + len(tag)
    if answer_index < 0:
        return INVALID_ANS

    numbers = ANS_RE.findall(completion[answer_index:])
    return numbers[-1] if numbers else INVALID_ANS


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

    def __init__(
        self,
        config,
        math_problems,
        llm,
        tokenizer,
        task_desc_str: str = None,
        cot_example_str: str = None,
        problem_format_str: str = None,
        reset=True,
        action_distribution_temperature=1.0,
        reasoning_effort: str = 'medium',
    ):
        super().__init__(
            config,
            math_problems,
            llm,
            tokenizer,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
            action_distribution_temperature,
            reasoning_effort,
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

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0
