import torch
from pathlib import Path

from .trainers import MultiAgentTrainer, SingleAgentTrainer
from .gym_environments import GymContinuousEnvMgr
from .unity_environments import UnityEnvMgr
from .factory import TrainerFactory
from .ddpg_agent import DDPGAgent
from .config import (
    ENV_TYPE,
    CLOUD,
    BUFFER_SIZE,
    BATCH_SIZE,
    N_EPISODES,
    MAX_T,
    N_WORKERS,
    MAX_WORKERS,
    LEARN_F,
    GAMMA,
    TAU,
    LR_ACTOR,
    LR_CRITIC,
    WEIGHT_DECAY,
    WINDOW_LEN,
)


def run_init():
    """Run main function from init reads init parameters from config.py"""
    if ENV_TYPE.lower() == "gym":
        envh = GymContinuousEnvMgr("Pendulum-v0")
        root_name = "gym"
        Trainer = SingleAgentTrainer
        upper_bound = 2.0
        solved = -250
    else:
        root = Path(__file__)
        if N_WORKERS == 1:
            envh_param = root / "envs/Reacher_Windows_x86_64-one-agent/Reacher.exe"
            root_name = "multi"
        else:
            envh_param = root / "envs/Reacher_Windows_x86_64-twenty-agents/Reacher.exe"
            root_name = "single"

        if not envh_param.exists():
            raise FileNotFoundError(
                f"Cannot file file: {envh_param} - check your file location and reacher_app.py"
            )
        envh = UnityEnvMgr(envh_param)
        Trainer = MultiAgentTrainer
        upper_bound = 1.0
        solved = 30.0

    if CLOUD:
        if N_WORKERS == 1:
            envh_param = Path(
                "/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64"
            )
        else:
            envh_param = Path("/data/Reacher_Linux_NoVis/Reacher.x86_64")
        if not envh_param.exists():
            raise FileNotFoundError(
                f"Cannot file file: {envh_param} - check your file location and reacher_app.py"
            )
        envh = UnityEnvMgr(envh_param)
    envh.start()
    state_size = envh.state_size
    action_size = envh.action_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = DDPGAgent(
        state_size=state_size,
        action_size=action_size,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        learn_f=LEARN_F,
        weight_decay=WEIGHT_DECAY,
        device=device,
        random_seed=42,
        upper_bound=upper_bound,
    )
    trainer = Trainer(
        agent=agent,
        env=envh,
        n_episodes=N_EPISODES,
        max_t=MAX_T,
        window_len=WINDOW_LEN,
        solved=solved,
        n_workers=N_WORKERS,
        max_workers=MAX_WORKERS,
        save_root=root_name,
    )
    scores = trainer.train(save_all=True)
    return scores


def run_loaded(
    trainer_file: str,
    agent_file: str,
    actor_file: str,
    critic_file: str,
):
    """
    Run main function from a loaded module
    Parameters
    ----------
    trainer_file : str
        filepath to toml file containing trainer hyperparameters
    agent_file : str
        filepath to toml file containing agent hyperparameters
    actor_file : str
        filepath to pth file containing actor weights
    critic_file : str
        filepath to pth file containing critic weights
    """
    if ENV_TYPE.lower() == "gym":
        #     scenarios = {'LunarLanderContinuous-v2',
        #                  'BipedalWalker-v3',
        #                  'Pendulum-v0'}
        envh_class = GymContinuousEnvMgr
        envh_param = "Pendulum-v0"
        trainer_class = SingleAgentTrainer
    else:
        root = Path(__file__)
        if N_WORKERS == 1:
            envh_param = root / "envs/Reacher_Windows_x86_64-one-agent/Reacher.exe"
        else:
            envh_param = root / "envs/Reacher_Windows_x86_64-twenty-agents/Reacher.exe"
        if not envh_param.exists():
            raise FileNotFoundError(
                f"Cannot file file: {envh_param} - check your file location and reacher_app.py"
            )
        envh_class = UnityEnvMgr
        trainer_class = MultiAgentTrainer
    if CLOUD:
        if N_WORKERS == 1:
            envh_param = Path(
                "/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64"
            )
        else:
            envh_param = Path("/data/Reacher_Linux_NoVis/Reacher.x86_64")
        if not envh_param.exists():
            raise FileNotFoundError(
                f"Cannot file file: {envh_param} - check your file location and reacher_app.py"
            )
    agent_class = DDPGAgent
    factory = TrainerFactory()
    trainer = factory(
        trainer_class,
        trainer_file,
        envh_class,
        envh_param,
        agent_class,
        agent_file,
        actor_file,
        critic_file,
    )

    scores = trainer.train(save_all=True)
    return scores


def main(
    trainer_file: str = None,
    agent_file: str = None,
    actor_file: str = None,
    critic_file: str = None,
):
    """
    Run Main - determines from loaded or from init if fields are populated
    """
    files = (trainer_file, agent_file, actor_file, critic_file)
    if any([file is None for file in files]):
        print("Hyperparameter Files not specified, running based on config.py")
        return run_init()
    print("Hyperparameter Files specified - running from loaded values")
    return run_loaded(*files)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DDPGAgent within an Environment, Pendulum or Reacher"
    )
    parser.add_argument(
        "--trainer",
        "-t",
        type=str,
        help="filepath to trainer_file hyperparameter file",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--agent",
        "-A",
        type=str,
        help="filepath to agent_file hyperparameter file",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--actor",
        "-a",
        type=str,
        help="filepath to actor_file hyperparameter file",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--critic",
        "-c",
        type=str,
        help="filepath to critic_file hyperparameter file",
        default=None,
        required=False,
    )

    args = parser.parse_args()
    main(args.trainer, args.agent, args.actor, args.critic)
