import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import time
import os
import shutil
import io
import sys
import json
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import glob

import grid2op
from grid2op.dtypes import dt_int
from grid2op.utils import ScoreL2RPN2022
from grid2op.Chronics import ChangeNothing
from grid2op.Agent import BaseAgent
from grid2op.Reward import BaseReward, RedispReward, L2RPNWCCI2022ScoreFun
from grid2op.Opponent import BaseOpponent
from grid2op.Episode import EpisodeReplay

SUBMISSION_DIR_ERR = """
ERROR: Impossible to find a "submission" package.
Agents should be included in a "submission" directory
A module with a function "make_agent" to load the agent that will be assessed."
"""

MAKE_AGENT_ERR = """
ERROR:  We could NOT find a function name \"make_agent\"
in your \"submission\" package. "
We remind you that this function is mandatory and should have the signature:
 make_agent(environment, path_agent) -> agent 

 - The "agent" is the agent that will be tested.
 - The "environment" is a valid environment provided.
   It will NOT be updated during the scoring (no data are fed to it).
 - The "path_agent" is the path where your agent is located
"""

ENV_TEMPLATE_ERR = """
ERROR: There is no powergrid found for making the template environment. 
Or creating the template environment failed.
The agent will not be created and this will fail.
"""

MAKE_AGENT_ERR2 = """
ERROR: "make_agent" is present in your package, but can NOT be used.

We remind you that this function is mandatory and should have the signature:
 make_agent(environment, path_agent) -> agent

 - The "agent" is the agent that will be tested.
 - The "environment" is a valid environment provided.
   It will NOT be updated during the scoring (no data are fed to it).
 - The "path_agent" is the path where your agent is located
"""

BASEAGENT_ERR = """
ERROR: The "submitted_agent" provided should be a valid Agent. 
It should be of class that inherit "BaseAgent" (`from grid2op.Agent import BaseAgent`) base class
"""

INFO_CUSTOM_REWARD = """
INFO: No custom reward for the assessment of your agent will be used.
"""

REWARD_ERR = """
ERROR: The "training_reward" provided should be a class.
NOT a instance of a class
"""

REWARD_ERR2 = """
ERROR: The "training_reward" provided is invalid.
It should inherit the "grid2op.Reward.BaseReward" class
"""

INFO_CUSTOM_OTHER = """
INFO: No custom other_rewards for the assessment of your agent will be used.
"""

KEY_OVERLOAD_REWARD = """
WARNING: You provided the key "{0}" in the "other_reward" dictionnary. 
This will be replaced by the score of the competition, as stated in the rules. Your "{0}" key WILL BE erased by this operation.
"""

KEY_OVERLOAD_WARN = """
The key "{}" cannot be used as a custom reward. It is used internally to get compute your score.
It will be disabled and erased.
"""

BACKEND_WARN = """
WARNING: Could not load lightsim2grid.LightSimBackend, falling back on PandaPowerBackend
"""

STARTING_THE_EVALUATION = """Starting the evaluation of your agent on the L2RPN_2022 dataset"""
ENDING_THE_EVALUATION = """Ending the evaluation of your agent on the L2RPN_2022 dataset"""

def write_gif(output_dir, agent_path, episode_name, start_step, end_step):
    try:
        epr = EpisodeReplay(agent_path)
        epr.replay_episode(episode_name,
                           fps=2.0,
                           display=False,
                           gif_name=episode_name,
                           start_step=start_step,
                           end_step=end_step,
                           load_info=None,
                           gen_info=None,
                           line_info=None
                           )
        gif_genpath = os.path.join(agent_path, episode_name,
                                   episode_name + ".gif")
        gif_outpath = os.path.join(output_dir, episode_name + ".gif")
        print(gif_genpath, gif_outpath)
        if os.path.exists(gif_genpath):
            shutil.move(gif_genpath, gif_outpath)
    except Exception as exc_:
        print("Cannot create GIF export with error \n{}".format(exc_))

def evaluate_agent(agent,save_path=False,gif_episode=None,gif_start=0,gif_end=50,cleanup=False,verbose=False):
    """
    :param agent: Agent to evaluate, The agent must have been created using the L2RPN_2022 env
    :param input_dir: Path to the dataset to create the environment
    :param output_dir: Path to the runner logs output dir
    :param gif_episode: Name of the episode to generate a gif for
    :param gif_start: int, Start step for gif generation
    :param gif_end: int, End step for gif generation
    :param cleanup: Cleanup runner logs
    :param verbose: Verbose runner output
    :return:
    """
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "L2RPN2022_Data")
    with open(os.path.join(input_dir, "L2RPN2022_config.json"), "r") as f:
        config = json.load(f)

    # create output dir if not existing
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log.debug("input dir: {}".format(input_dir))
    log.debug("output dir: {}".format(save_path))

    try:
        from lightsim2grid import LightSimBackend
        backend_cls = LightSimBackend
    except:
        print(BACKEND_WARN)
        from grid2op.Backend import PandaPowerBackend
        backend_cls = PandaPowerBackend

    # create the agent
    try:
        submitted_agent = agent
    except Exception as exc_:
        print(MAKE_AGENT_ERR2)
        print("The error was: {}".format(exc_))
        raise RuntimeError(MAKE_AGENT_ERR2) from exc_

    if not isinstance(submitted_agent, BaseAgent):
        print(BASEAGENT_ERR)
        raise RuntimeError(BASEAGENT_ERR)

    # import the rewards and other things
    other_rewards = {}

    # add the other rewards to compute the real score
    key_score = config.get("score_config",{}).get("key_score",{})
    if key_score in other_rewards:
        print(KEY_OVERLOAD_WARN.format(key_score))
    other_rewards[key_score] = L2RPNWCCI2022ScoreFun

    # create the real environment
    real_env = grid2op.make(input_dir,
                            reward_class=RedispReward,
                            other_rewards=other_rewards,
                            backend=backend_cls(),
                            )

    # this is called after, so that no one can change this sequence
    np.random.seed(int(config["score_config"]["seed"]))
    max_int = np.iinfo(dt_int).max
    # env seeds are read from the json
    env_seeds = [int(config["episodes_info"][os.path.split(el)[-1]]["seed"]) for el in
                 sorted(real_env.chronics_handler.real_data.subpaths)]
    # agent seeds are generated with the provided random seed
    agent_seeds = list(np.random.randint(max_int, size=int(config["nb_scenario"])))
    path_save = os.path.abspath(save_path)
    scores = ScoreL2RPN2022(env=real_env,
                            env_seeds=env_seeds,
                            agent_seeds=agent_seeds,
                            nb_scenario=int(config["nb_scenario"]),
                            min_losses_ratio=float(config["score_config"]["min_losses_ratio"]),
                            verbose=0 if not verbose else 2,
                            max_step=-1,
                            nb_process_stats=1)
    print(STARTING_THE_EVALUATION)
    beg_ = time.perf_counter()
    scores, n_played, total_ts = scores.get(submitted_agent, path_save=path_save, nb_process=1)
    res_scores = {"scores": [float(score) for score in scores],
                  "n_played": [int(el) for el in n_played],
                  "total_ts": [int(el) for el in total_ts]}
    end_ = time.perf_counter()
    print(f"[INFO] agent scoring time time: {end_ - beg_:.2f}s")
    print(ENDING_THE_EVALUATION)
    with open(os.path.join(path_save, "res_agent.json"), "w", encoding="utf-8") as f:
        json.dump(obj=res_scores, fp=f)

    if gif_episode is not None:
        beg_ = time.perf_counter()
        gif_input = os.path.join(save_path)
        write_gif(save_path, gif_input, gif_episode,
                  gif_start, gif_end)
        end_ = time.perf_counter()
        print(f"[INFO] gif writing time: {end_ - beg_:.2f}s")
    real_env.close()

    if cleanup:
        cmds = [
            "find {} -name '*.npz' | xargs -i rm -rf {}",
            "find {} -name 'dict_*.json' | xargs -i rm -rf {}",
            "find {} -name '_parameters.json' | xargs -i rm -rf {}"
        ]
        for cmd in cmds:
            os.system(cmd.format(save_path, "{}"))
    print("Done and data saved in : \"{}\"".format(path_save))
    process_scores(agent_dir=save_path)

SCORE_TXT = "scores.txt"
ALL_SCORE_CSV = "score_episodes.csv"
RESULT_HTML = "results.html"
META_JSON = "episode_meta.json"
TIME_JSON = "episode_times.json"
REWARD_JSON = "other_rewards.json"
SCORES_JSON = "res_agent.json"
MIN_SCORE = -130.
TIMEOUT_TIME = 5000.

def create_fig(title,
               x=4,
               y=2,
               width=1280,
               height=4 * 720,
               dpi=96):
    w = width / dpi
    h = height / dpi
    fig, axs = plt.subplots(ncols=x, nrows=y, figsize=(w, h), sharey=True)
    fig.suptitle(title)
    return fig, axs

def draw_steps_fig(ax, step_data, ep_score, ep_share):
    nb_timestep_played = int(step_data["nb_timestep_played"])
    chronics_max_timestep = int(step_data["chronics_max_timestep"])
    n_blackout_steps = chronics_max_timestep - nb_timestep_played
    title_fmt = "{}\n({:.2f}/{:.2f})"
    scenario_dir = os.path.basename(step_data["chronics_path"])
    scenario_name = title_fmt.format(scenario_dir, ep_score, ep_share)
    labels = 'Played', 'Blackout'
    fracs = [
        nb_timestep_played,
        n_blackout_steps
    ]

    def pct_fn(pct):
        n_steps = int(pct * 0.01 * chronics_max_timestep)
        return "{:.1f}%\n({:d})".format(pct, n_steps)

    ax.pie(fracs, labels=labels,
           autopct=pct_fn,
           startangle=90.0)
    ax.set_title(scenario_name)

def draw_rewards_fig(ax, reward_data, step_data):
    nb_timestep_played = int(step_data["nb_timestep_played"])
    chronics_max_timestep = int(step_data["chronics_max_timestep"])
    n_blackout_steps = chronics_max_timestep - nb_timestep_played
    scenario_name = "Scenario " + os.path.basename(step_data["chronics_path"])

    x = list(range(chronics_max_timestep))
    n_rewards = len(reward_data[0].keys())
    y = [[] for _ in range(n_rewards)]
    labels = list(reward_data[0].keys())
    for rel in reward_data:
        for i, v in enumerate(rel.values()):
            y[i].append(v)
    for i in range(n_rewards):
        y[i] += [0.0] * n_blackout_steps

    for i in range(n_rewards):
        ax.plot(x, y[i], label=labels[i])
    ax.set_title(scenario_name)
    ax.legend()

def fig_to_b64(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    fig_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return fig_b64

def html_result(score, duration, fig_list):
    html = """<html><head></head><body>\n"""
    html += """<div style='margin: 0 auto; width: 500px;'>"""
    html += """<p>Score {}</p>""".format(np.round(score, 3))
    html += """<p>Duration {}</p>""".format(np.round(duration, 2))
    html += """</div>"""
    for i, figure in enumerate(fig_list):
        html += '<img src="data:image/png;base64,{0}"><br>'.format(figure)
    html += """</body></html>"""
    return html

def html_error():
    html = """<html><head></head><body>\n"""
    html += """Invalid submission"""
    html += """</body></html>"""
    return html

def write_output(output_dir, html_content, episode_score_dic, duration, save_all_score=False):
    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Write scores
    score_filename = os.path.join(output_dir, SCORE_TXT)
    with open(score_filename, 'w') as f:
        f.write("score: {:.6f}\n".format(episode_score_dic["total"]))
        f.write("duration: {:.6f}\n".format(duration))

    if save_all_score:
        print(f"\t\t saving all scores")
        all_score_filename = os.path.join(output_dir, ALL_SCORE_CSV)
        global_keys = {"total", "total_operation", "total_attention"}
        episode_score_df = {'scenario': [k for k, val in episode_score_dic.items() if k not in global_keys],
                            'score': [float(val) for k, val in episode_score_dic.items() if k not in global_keys],
                            }
        episode_score_df["scenario"].append("global")
        episode_score_df["score"].append(episode_score_dic["total"])
        all_score_pd = pd.DataFrame(episode_score_df).round(1)
        all_score_pd.to_csv(all_score_filename, index=False)

    # Write results
    result_filename = os.path.join(output_dir, RESULT_HTML)
    with open(result_filename, 'w') as f:
        f.write(html_content)


def process_scores(agent_dir):
    DEFAULT_TIMEOUT_SECONDS = 20 * 60

    save_all_score = True

    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "L2RPN2022_Data")
    with open(os.path.join(input_dir, "L2RPN2022_config.json"), "r") as f:
        config = json.load(f)

    # Fail if input doesn't exists
    if not os.path.exists(agent_dir):
        error_score = (MIN_SCORE, -100., -200.)
        error_duration = DEFAULT_TIMEOUT_SECONDS + 1
        write_output(agent_dir, html_error(), error_score, error_duration)
        sys.exit("Your submission is not valid.")

    # Create output variables
    total_duration = 0.0
    total_score = 0.0

    ## Create output figures
    step_w = 4
    nb_episode = config['nb_scenario']
    timeout_time = float(config["score_config"]["timeout"])
    step_h = max(nb_episode // step_w, 1)
    if nb_episode % step_w > 0:
        step_h += 1
    step_fig, step_axs = create_fig("Completion", x=step_w, y=step_h)
    reward_w = 2
    reward_h = max(nb_episode // reward_w, 1)
    reward_title = "Cost of grid operation & Custom rewards"
    reward_fig, reward_axs = create_fig(reward_title,
                                        x=reward_w, y=reward_h,
                                        height=3500)
    episode_index = 0
    episode_names = config["episodes_info"].keys()
    score_config = config["score_config"]

    episode_score_dic = {}
    scores_json = os.path.join(agent_dir, SCORES_JSON)
    if not os.path.exists(scores_json):
        # the score has not been computed
        total_duration = 99999
        total_score = MIN_SCORE
    else:
        with open(scores_json, "r", encoding="utf-8") as f:
            dict_scores = json.load(fp=f)

        # I loop through all the episodes
        for episode_id in sorted(episode_names):
            # Get info from config
            episode_info = config["episodes_info"][episode_id]
            episode_len = float(episode_info["length"])
            episode_weight = episode_len / float(score_config["total_timesteps"])

            # Compute episode files paths
            scenario_dir = os.path.join(agent_dir, episode_id)
            meta_json = os.path.join(scenario_dir, META_JSON)
            time_json = os.path.join(scenario_dir, TIME_JSON)
            reward_json = os.path.join(scenario_dir, REWARD_JSON)
            if not os.path.isdir(scenario_dir) or \
                    not os.path.exists(meta_json) or \
                    not os.path.exists(time_json) or \
                    not os.path.exists(reward_json) or \
                    not os.path.exists(scores_json):
                episode_score = MIN_SCORE
            else:
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                with open(reward_json, "r", encoding="utf-8") as f:
                    other_rewards = json.load(f)

                with open(time_json, "r", encoding="utf-8") as f:
                    timings = json.load(f)

                episode_score = dict_scores["scores"][episode_index]  # for the total score
                episode_score_dic[episode_id] = dict_scores["scores"][episode_index]  # for all the scores

                # Draw figs
                step_ax_x = episode_index % step_w
                step_ax_y = episode_index // step_w
                draw_steps_fig(step_axs[step_ax_y, step_ax_x],
                               meta, episode_score * episode_weight,
                               episode_weight * 100.0)
                reward_ax_x = episode_index % reward_w
                reward_ax_y = episode_index // reward_w
                draw_rewards_fig(reward_axs[reward_ax_y, reward_ax_x],
                                 other_rewards, meta)

            # Sum durations and scores
            total_duration += float(timings["Agent"]["total"])
            total_score += episode_weight * episode_score

            # Loop to next episode
            episode_index += 1

            if total_duration >= timeout_time:
                # terminate everything is the timeout is reached
                total_duration = TIMEOUT_TIME
                total_score = MIN_SCORE
                break

    episode_score_dic["total"] = total_score

    # Format result html page
    step_figb64 = fig_to_b64(step_fig)
    reward_figb64 = fig_to_b64(reward_fig)
    html_out = html_result(total_score,
                           total_duration,
                           fig_list=[step_figb64, reward_figb64])

    # Write final output
    print(f"\t\t: {episode_score_dic}")
    write_output(agent_dir, html_out, episode_score_dic, total_duration, save_all_score)
    try:
        # Copy gifs if any
        gif_input = os.path.abspath(input_dir)
        gif_output = os.path.abspath(agent_dir)
        gif_names = glob.glob(os.path.join(gif_input, "*.gif"))
        if gif_names:
            for g_n in gif_names:
                g_n_local = os.path.split(g_n)[-1]
                shutil.copy(os.path.join(g_n),
                            os.path.join(gif_output, g_n_local))
        # gif_cmd = "find {} -name '*.gif' | xargs -i cp {} {}"
        # os.system(gif_cmd.format(gif_input, "{}", gif_output))
    except Exception as exc_:
        print(f"\t\t WARNING: GIF copy failed, no gif will be available. Error was: {exc_}")

    with open(os.path.join(agent_dir, "scores.txt"), "r") as f:
        scores = f.readlines()
    scores = [el.rstrip().lstrip().split(":") for el in scores]
    print("Your scores are :")
    res = pd.DataFrame(scores)
    log.info('Your scores are :\n\t' + res.to_string().replace('\n', '\n\t'))
    print(res)