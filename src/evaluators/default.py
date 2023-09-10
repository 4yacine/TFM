import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import os, json
import pandas as pd

import matplotlib.pyplot as plt


def plot_score(score_list, ylabel="Reward", xlabel="Epoch (Episode)", title="", marker=".", linestyle="None",save_path=False):
    plt.plot(score_list, marker=marker, linestyle=linestyle)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path,F'{title}.png'))
        plt.clf()
        plt.close()
    else:
        plt.show()



def plot_score_dict(score_dict,save_path=False):
    sclist = []
    eplist = []
    for idx in score_dict:
        sclist.append(score_dict.get(idx).get("score"))
        eplist.append(score_dict.get(idx).get("steps"))
    plot_score(sclist, title=F"Episode scores ",save_path=save_path)
    plot_score(eplist, title=F"Episode length",ylabel="Steps",save_path=save_path)


def test_and_evaluate_agent(env, agent, episodes=100, max_steps=10000,
                            plot_scores=False,save_path=False,verbose=False,render_step=False):

    scores = {}  # puntuaciones de cada episodio

    for i_episode in range(0, episodes):
        state = env.reset()
        score = 0
        reward = 0
        done = False

        for t in range(1,max_steps+1):
            action = agent.act(state, reward, done)
            #TODO: make the render dependant of the environment used. grid2op is state.render but anm is env.render
            #if render_step:
            #    _ = state.render()
            state, reward, done, info = env.step(action)
            if verbose > 2:
                print(F"step: {t} \t-reward: {reward} \t-done: {done}")
            score += reward
            if done:
                if verbose:
                    print(F"DONE == TRUE after {t} steps. Score: {score}")
                if verbose>1:
                    for item in info:
                        print(F"={item}=")
                        print(info.get(item))
                    print("==STATE==")
                    print(state)
                    print("==Action==")
                    print(action)
                    print("==Done==")
                break
        scores[F"{i_episode}"] = {"score": score, "steps": t}
    log.debug(scores)
    df = pd.DataFrame.from_dict(scores,orient='index')
    log.info(F"Mean Scores over [{episodes} episodes] : \n{df.mean()}")
    print(F"Mean score over [{episodes} episodes] : \n{df.mean()}")
    if plot_scores:
        plot_score_dict(scores,save_path)
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(F'{os.path.join(save_path,"scores.json")}', 'w') as scoreFile:
            print(json.dumps(scores, indent=4), file=scoreFile)
        with open(F'{os.path.join(save_path,"meanscores.pd")}', 'w') as scoreFile:
            print(df.mean(), file=scoreFile)

    return scores


def pandas_from_score_dict(score_dict):
    jsons_data = pd.DataFrame(columns=['score', 'steps'])

    for index in score_dict:
        dat = score_dict.get(index)

        score = dat.get("score")
        steps = dat.get("steps")

        jsons_data.loc[index] = [score, steps]
    return jsons_data


def get_mean_of_score_dict(score_dict):
    return pandas_from_score_dict(score_dict)["score"].mean()


if __name__ == "__main__":

    path_to_json = './scores'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

    dataframes = {}
    for file_index, js in enumerate(json_files):
        jsons_data = pd.DataFrame(columns=['Agent', 'Score (Reward sum before failure)', 'Steps before failure'])
        agent_name = js.rpartition("_scores")[0]
        with open(os.path.join(path_to_json, js)) as json_file:

            json_text = json.load(json_file)
            for index in json_text:
                dat = json_text.get(index)
                # here you need to know the layout of your json and each json has to have
                # the same structure (obviously not the structure I have here)
                score = dat.get("score")
                steps = dat.get("steps")
                # here I push a list of data into a pandas DataFrame at row given by 'index'
                jsons_data.loc[index] = [agent_name, score, steps]
        dataframes[agent_name] = jsons_data

    Metrics = pd.DataFrame(columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

    for dfname in dataframes:
        df = dataframes.get(dfname)

        Metrics.loc[F"{dfname} score"] = df['Score (Reward sum before failure)'].describe().tolist()
        Metrics.loc[F"{dfname} steps"] = df['Steps before failure'].describe().tolist()

    print(Metrics)

    pd.set_option('display.max_columns', None)
    dataframe = pd.concat([dataframes.get(dfname) for dfname in dataframes], ignore_index=True, sort=False)
    dataframe.boxplot(column=["Steps before failure"], by="Agent", meanline=True, showmeans=True)
    dataframe.boxplot(column=["Score (Reward sum before failure)"], by="Agent", meanline=True, showmeans=True)
    plt.show()
