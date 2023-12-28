from minatar import Environment

from q3_schedule import LinearExploration, LinearSchedule
from q4_linear_torch import Linear

from configs.q6_train_atari_linear import config
from utils.general import export_mean_plot
import logging
 
if __name__ == "__main__":
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # disable font manager warnings
    # make env
    env = Environment("breakout")
    num_runs = 3

    for i in range(num_runs):
        # exploration strategy
        exp_schedule = LinearExploration(
            env, config.eps_begin, config.eps_end, config.eps_nsteps
        )

        # learning rate schedule
        lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

        # train model
        model = Linear(env, config)
        model.run(exp_schedule, lr_schedule, run_idx=i + 1)

    export_mean_plot("Scores", config.plot_output, config.output_path)
