'''
'''

EVALUATION_DEFAULT="DEFAULT"
EVALUATION_L2RPN="L2RPN2022"
EVALUATION_L2RPN2022="L2RPN2022"
EVALUATION_GRID2OP="GRID2OP_BASELINE"
#EVALUATION_L2RPN_CASE14 = "L2RPNCASE14"
EVALUATION_CITYLEARN_ENV="EVALUATION_CITYLEARN"
TRAINING_DEFAULT="DEFAULT"

ALL_ATTR_OBS = (
    "year",
    "month",
    "day",
    "hour_of_day",
    "minute_of_hour",
    "day_of_week",
    "gen_p",
    "gen_p_before_curtail",
    "gen_q",
    "gen_v",
    "gen_margin_up",
    "gen_margin_down",
    "load_p",
    "load_q",
    "load_v",
    "p_or",
    "q_or",
    "v_or",
    "a_or",
    "p_ex",
    "q_ex",
    "v_ex",
    "a_ex",
    "rho",
    "line_status",
    "timestep_overflow",
    "topo_vect",
    "time_before_cooldown_line",
    "time_before_cooldown_sub",
    "time_next_maintenance",
    "duration_next_maintenance",
    "target_dispatch",
    "actual_dispatch",
    "storage_charge",
    "storage_power_target",
    "storage_power",
    "curtailment",
    "curtailment_limit",
    "curtailment_limit_effective",
    "thermal_limit",
    "is_alarm_illegal",
    "time_since_last_alarm",
    "last_alarm",
    "attention_budget",
    "was_alarm_used_after_game_over",
    "current_step",
    "max_step",
    "theta_or",
    "theta_ex",
    "load_theta",
    "gen_theta",
)


ALL_ATTR_ACT = (
    "set_line_status",
    "change_line_status",
    "set_bus",
    "change_bus",
    "redispatch",
    "set_storage",
    "curtail",
    "raise_alarm",
)

ALL_ATTR_ACT_DISCRETE = (
    "set_line_status",
    "set_line_status_simple",
    "change_line_status",
    "set_bus",
    "change_bus",
    "sub_set_bus",
    "sub_change_bus",
    "one_sub_set",
    "one_sub_change",
    "raise_alarm"
)
