from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import dspy
from dspy.predict.predict import Module, Prediction


class BestOfN(Module):
    def __init__(
        self,
        module: Module,
        N: int,  # noqa: N803
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ):
        """
        Runs a module up to `N` times with different rollout IDs at `temperature=1.0` and
        returns the best prediction out of `N` attempts or the first prediction that passes the
        `threshold`.

        Args:
            module (Module): The module to run.
            N (int): The number of times to run the module.
            reward_fn (Callable[[dict, Prediction], float]): The reward function which takes in the args passed to the module, the resulting prediction, and returns a scalar reward.
            threshold (float): The threshold for the reward function.
            fail_count (Optional[int], optional): The number of times the module can fail before raising an error. Defaults to N if not provided.

        Example:
            ```python
            import dspy

            dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

            # Define a QA module with chain of thought
            qa = dspy.ChainOfThought("question -> answer")

            # Define a reward function that checks for one-word answers
            def one_word_answer(args, pred):
                return 1.0 if len(pred.answer.split()) == 1 else 0.0

            # Create a refined module that tries up to 3 times
            best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)

            # Use the refined module
            result = best_of_3(question="What is the capital of Belgium?").answer
            # Returns: Brussels
            ```
        """
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N  # default to N if fail_count is not provided

    def _run_single(self, rid, kwargs):
        """Run a single attempt with the given rollout id."""
        lm = self.module.get_lm() or dspy.settings.lm
        lm_ = lm.copy(rollout_id=rid, temperature=1.0)
        mod = self.module.deepcopy()
        mod.set_lm(lm_)

        with dspy.context(trace=[]):
            pred = mod(**kwargs)
            trace = dspy.settings.trace.copy()
            # NOTE: Not including the trace of reward_fn.
            reward = self.reward_fn(kwargs, pred)

        return pred, trace, reward

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        rollout_ids = [start + i for i in range(self.N)]
        best_pred, best_trace, best_reward = None, None, -float("inf")
        fail_count = 0

        with ThreadPoolExecutor(max_workers=self.N) as executor:
            future_to_rid = {
                executor.submit(self._run_single, rid, kwargs): rid
                for rid in rollout_ids
            }

            for future in as_completed(future_to_rid):
                rid = future_to_rid[future]
                try:
                    pred, trace, reward = future.result()

                    if reward > best_reward:
                        best_reward, best_pred, best_trace = reward, pred, trace

                    # Return immediately if threshold is met
                    if reward >= self.threshold:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                except Exception as e:
                    fail_count += 1
                    print(f"BestOfN: Attempt with rollout id {rid} failed: {e}")
                    if fail_count > self.fail_count:
                        raise e

        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_pred, best_reward
