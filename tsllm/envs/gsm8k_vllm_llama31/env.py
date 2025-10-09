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
ANS_RE = re.compile(r"The final answer is (\-?[0-9\.\,]+)")
STOP_STR = "The final answer is "
QUESTION_KEY = "question"
PROBLEM_FORMAT_STR = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

There are 15 trees originally.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6.
The final answer is 6<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

There are originally 3 cars.
2 more cars arrive.
3 + 2 = 5.
The final answer is 5<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Originally, Leah had 32 chocolates.
Her sister had 42.
So in total they had 32 + 42 = 74.
After eating 35, they had 74 - 35 = 39.
The final answer is 39<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Jason started with 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 8.
The final answer is 8<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Shawn started with 5 toys.
If he got 2 toys each from his mom and dad, then that is 4 more toys.
5 + 4 = 9.
The final answer is 9<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

There were originally 9 computers.
For each of 4 days, 5 more computers were added.
So 5 * 4 = 20 computers were added.
9 + 20 is 29.
The final answer is 29<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Michael started with 58 golf balls.
After losing 23 on tuesday, he had 58 - 23 = 35.
After losing 2 more, he had 35 - 2 = 33 golf balls.
The final answer is 33<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Olivia had 23 dollars.
5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
So she has 23 - 15 dollars left.
23 - 15 is 8.
The final answer is 8<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following problem, reason and give a final answer to the problem.
Problem: {question}
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
COT_EXAMPLES = None
COT_TASK_DESC = None
SEP = None
SEP_TOKENS = ['.Ċ']
LAST_QUERY_TOKEN = None


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

    sep: str = SEP
    sep_tokens: list = SEP_TOKENS
    last_query_token: str = LAST_QUERY_TOKEN

    @staticmethod
    def build_response_str(
            answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool
    ):
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
        return problem_format_str.format(question=problem_input)

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
            task_desc_str: str = COT_TASK_DESC,
            cot_example_str: str = COT_EXAMPLES,
            problem_format_str: str = PROBLEM_FORMAT_STR,
            reset=True,
            action_distribution_temperature=1.0,
            reasoning_effort='medium',
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

        self.last_query_token_id = self.tokenizer.encode(self.last_query_token)[0]
        self.sep_token_ids = self.tokenizer.convert_tokens_to_ids(self.sep_tokens)
        stop_token_ids = [self.tokenizer.eos_token_id] + self.sep_token_ids

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

    def get_state(self):
        return ''.join(self.action_history)

    def init_action_history(self):
        # add the first prompted questions
        question = self.math_problem["question"]
        return [self.build_query_str(cot_task_desc=None, cot_examples=None, problem_format_str=self._problem_format_str,
                                     problem_input=question, sep=None, )]

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

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    @property
    def question(self):
        return self.action_history[0]

    @property
    def answer(self):
        return ''.join(self.action_history[1:])

    def get_done_and_info(self):
        info = {"winner": 0}
        # done when reaches maximum length or LLM generates stop words
        terminated = len(self.action_history) > 1 and self.stop_str in self.action_history[-1]
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
    sep_tokens: list = SEP_TOKENS
    last_query_token = LAST_QUERY_TOKEN

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
