from gymnasium.envs.registration import register

register(
     id="ChessVsSelf-v0",
     entry_point="lib.gymchess.ChessEnv:ChessEnv",
     max_episode_steps=1500,
)