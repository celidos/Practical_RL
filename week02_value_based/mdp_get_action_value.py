
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    q = 0.0
    for s_new, proba in mdp.get_next_states(state, action).items():
        q += proba * (mdp.get_reward(state, action, s_new) + \
                      gamma * state_values[s_new])

    return q
