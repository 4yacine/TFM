import logging
import os
import inspect

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from src import constants

def evaluate_agent(agent,experiment_config,eval_override=None):
    #Get evaluation method
    evaluation = experiment_config.get("evaluation")
    if eval_override is not None:
        evaluation = eval_override.get("evaluation")

    # Get evaluation kwargs
    evaluation_kwargs = experiment_config.get("evaluation_kwargs",{})
    if eval_override is not None:
        evaluation_kwargs = eval_override.get("evaluation_kwargs",{})

    #if evaluation method is a list we call evaluate_agent for each element of the list
    if  isinstance(evaluation,list):
        returns = []
        for eval in evaluation:
            returns.append(evaluate_agent(agent,experiment_config,eval_override=eval))
        return returns

    save_path = evaluation_kwargs.get("save_path", None)
    global_save_path = experiment_config.get("save_path", None)
    experiment_name =  experiment_config.get('name')
    if save_path:
        evaluation_kwargs["save_path"] = os.path.join(save_path,experiment_name)
    elif global_save_path:
        evaluation_kwargs["save_path"] = os.path.join(os.path.join(global_save_path,experiment_name), experiment_name)

    if evaluation in [constants.EVALUATION_DEFAULT, True]:
        from src.evaluators import default
        return default.test_and_evaluate_agent(
            experiment_config.get("env").get("env"),
            agent,
            **evaluation_kwargs
        )

    elif evaluation in [constants.EVALUATION_GRID2OP, True]:
        from src.evaluators import baseline
        return baseline.evaluate(
            experiment_config.get("env").get("env"),
            agent,
            **evaluation_kwargs
        )

    elif evaluation == constants.EVALUATION_L2RPN2022:
        from src.evaluators import L2RPN2022
        return L2RPN2022.evaluate_agent(
            agent,
            **evaluation_kwargs)

    elif evaluation in [constants.EVALUATION_CITYLEARN_ENV]:
        from src.evaluators import citylearn
        return citylearn.evaluate_environment(
            experiment_config.get("env").get("env"),
            agent,
        )

    elif inspect.isclass(evaluation):
        log.warning(F"Class-base trainer are not fully designed")
        return evaluation(agent, **evaluation_kwargs).evaluate()

    elif inspect.isfunction(evaluation):
        return evaluation(agent, **evaluation_kwargs)

    else:
        log.warning(F"[{evaluation}] is not a valid evaluation method or class.")
