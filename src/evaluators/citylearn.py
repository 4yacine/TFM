from numpy import average

def evaluate_environment(env, agent,save_path=None):
    observations = env.reset()
    reward_hist = [] #Yacine
    while not env.done:
        actions = agent.predict(observations, deterministic=True)
        if len(actions) == 2:
            #sometimes predict return a touple...
            #dont know the reason
            actions,_ = actions

        # modificado por Yacine
        observations, reward, done, info = env.step(actions) 
        reward_hist.append(reward)

    for n, nd in env.evaluate().groupby('name'):
        nd = nd.pivot(index='name', columns='cost_function', values='value').round(3)
        print(n, ':', nd.to_dict('records'))

    # añadido por Yacine
    column_average_reward = average(reward_hist, axis=0)
    print("\nReward: ", column_average_reward)
    print("\nAverage reward", sum(column_average_reward)/len(column_average_reward))
    #print("\nReward: ", reward_hist)
    #print("\nDone: ", done)
    #print("\nInfo: ", info)