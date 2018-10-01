# ein einfach Q-lerning Beispiel
import numpy as np
import pandas as pa
import os

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1


class QLearningTable:
    def __init__(self, states, actions, fn):
        self.states = states
        self.actions = actions
        self.csvfn = fn
        self.init_table()

    # erstellt eine Q-Tabelle, wenn falls es keine gespeicherte Tabelle git
    # oder ladet die Tabelle aus einer gespeicherte Tabelle
    # die Zeilen sind State und die Spalten sind Aktion
    def init_table(self):
        if (os.path.isfile(self.csvfn)):
            self.table = pa.read_csv(self.csvfn,
                                     index_col='Employee',
                                     header=0,
                                     names=self.actions)
        else:
            self.table = pa.DataFrame(np.zeros((len(self.states), len(self.actions))), index=self.states,
                                      columns=self.actions)
    # sucht eine Aktion, die maxmale Q-Wert hat, durch die State oder eine zufällige Aktion
    def choose_action(self, state, other_state, islern=True):
        # print(state)
        n_state = str(state) + "_" + str(other_state)
        state_actions = self.table.loc[n_state]
        # print (state_actions)
        if ((np.random.uniform() > EPSILON and islern) or (state_actions == 0).all()):
            return np.random.choice(self.actions)
        else:
            return state_actions.idxmax()

    # aktuallisiert die Q-Tabelle mit der Belohnung
    def lern(self, state, other_state, result, next_s, next_other_s, action, reward):
        n_state = str(state) + "_" + str(other_state)
        q_predict = self.table.loc[n_state, action]

        if (result == "gameover" or reward != 0):
            q_target = reward
        else:
            n_next_s = str(next_s) + "_" + str(next_other_s)
            q_target = reward + GAMMA * self.table.loc[n_next_s, :].max()
        self.table.loc[n_state, action] += ALPHA * (q_target - q_predict)

    def get_table(self):
        return self.table


if __name__ == "__main__":
    states = ["11_12", "13,_21", "22_23", "31_33"]
    Actions = ["up", "right", "down", "left"]
    rl = QLearningTable(states, Actions)
    action = rl.choose_action(11, 12)
    print(action)
