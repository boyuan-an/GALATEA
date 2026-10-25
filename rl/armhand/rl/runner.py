import torch
from rl_games.torch_runner import Runner as _Runner

from armhand.rl.agent import BlockPPOAgent


class Runner(_Runner):
    def __init__(self, algo_observer=None):
        super().__init__(algo_observer)
        self.algo_factory.register_builder("block_ppo", lambda **kwargs: BlockPPOAgent(**kwargs))

    def run_train(self, args):
        agent = self.algo_factory.create(self.algo_name, base_name="run", params=self.params)
        checkpoint = args.get("checkpoint")
        if checkpoint:
            if args.get("checkpoint_model_only", False):
                agent.model.load_state_dict(torch.load(checkpoint, map_location=agent.ppo_device)["model"])
            else:
                agent.restore(checkpoint, args.get("from_ckpt_epoch", True))
        if args.get("frame0_eval_only", False):
            agent.init_tensors()
            agent.obs = agent.env_reset()
            if agent.global_rank == 0:
                print(f"[frame0_eval_only] epoch={agent.epoch_num} frame={agent.frame}", flush=True)
            agent.run_frame0_eval(agent.epoch_num, agent.frame)
            return
        agent.train()
