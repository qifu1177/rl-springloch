import numpy as np
from game_env import Game
import time

States = [11, 12, 21, 22, 23, 31] #die alle mögliche Positionen der Schachfigur
Actions = ["up", "right", "down", "left"] #die alle Bewegungsaktion
MAX_EPISODES = 10050 #die maximale Spiele
TEST_MAX_EPISODES = 10000 #die Spiele, nach den die graphische Darstellung gezeigt wird
MAX_ROUND = 50 #über die Anzahl wird die Spiel beendet
FRESH_TIME = 0.01 #wie lang dauert ein Schritt für die graphische Darstellung
ShowGrapie = True #ob die graphische Darstellung scheint

#erstellt die alle mögliche Kombinationen zwisch die beide Spieler
def create_all_state():
    list = []
    for a_state in States:
        for b_state in States:
            if (a_state != b_state):
                list.append(str(a_state) + "_" + str(b_state))
    return list

#druckt die letze Beziehung zwischen die State und Aktion
def create_actions_table(rl):
    for a_state in States:
        for b_state in States:
            if (a_state != b_state):
                rl.choose_action(a_state, b_state, False)

#gibt die nächste State, Belohnung und Ergibnis nach die aktuelle State und Acktion zurück
def get_env_feedback(player, state, other_state, action, used_actions):
    # This is how agent will interact with the environment
    next_state = update_state(state, action)
    reward = 0
    result = next_state
    next_other_state = other_state
    # falls die Schachfigur in der schwartz Löcher geht
    if (next_state in [13, 32, 33]):
        if (player == "A"):
            reward = -1
        else:
            reward = 1
        result = "gameover"
    # falls die andere Schachfigur kein Wahl hat
    elif ((next_state == 21 and other_state == 31) or (next_state == 22 and other_state == 23)):
        if (player == "A"):
            reward = 1
        else:
            reward = -1
        result = "gameover"
        # next_other_state = 32
    # falls die nächste State die Regeln verstößt
    elif (not check_station(next_state, other_state)):
        if (action not in used_actions):
            # print(used_actions)
            used_actions.append(action)
        if (player == "A"):
            reward = -1
            result = "gameover"
        else:
            action = get_next_action(used_actions)
            if (not action):
                return next_state, 0, "", "error", next_other_state
            else:
                return get_env_feedback(player, state, other_state, action, used_actions)

    return next_state, reward, action, result, next_other_state

#überprüft, ob die State die Regeln verstößt
def check_station(state, other_state):
    backv = True
    row = int(state / 10)
    col = state % 10
    if (state == other_state):
        backv = False
    elif (row < 1 or row > 3):
        backv = False
    elif (col < 1 or col > 3):
        backv = False
    return backv

# gibt andere Aktionsmöglichkeit für Spieler B
def get_next_action(used_actions):
    if (len(Actions) == len(used_actions)):
        return False
    action = np.random.choice(Actions)
    if (action in used_actions):
        return get_next_action(used_actions)
    else:
        return action

# zufällig setzt die Anfänge der Schachfigur
def random_getstart():
    a = np.random.choice(States)
    b = np.random.choice(States)
    if (a == b):
        return random_getstart()
    else:
        return a, b

# berechnt die nächste State nach die aktuelle State und Aktion
def update_state(state, action):
    row = int(state / 10)
    col = state % 10
    reward = 0

    if (action == "up"):
        row -= 1
    elif (action == "right"):
        col += 1
    elif (action == "down"):
        row += 1
    else:
        col -= 1

    next_state = row * 10 + col
    return next_state

# berechnet die Aktion für Spieler B
def get_b_action(a_state, b_state, used_actions):
    action = np.random.choice(Actions)
    next_state = update_state(b_state, action)
    if (next_state not in States or b_state == a_state):
        if (action not in used_actions):
            used_actions.append(action)
        if (len(used_actions) < len(Actions)):
            return get_b_action(a_state, b_state, used_actions)
    return action

# optimiert die Q-Netz durch die State und Aktion
def update(env, rl, start_episode=500, interval=50):
    sum = {"A": 0, "B": 0}
    for episode in range(MAX_EPISODES):
        step_counter = 0
        a_state, b_state = random_getstart()
        if (episode > TEST_MAX_EPISODES and ShowGrapie):
            env.step("A", a_state)
            env.render()
            time.sleep(FRESH_TIME)
            env.step("B", b_state)
            env.render()
            time.sleep(FRESH_TIME)

        win = ""
        print("start...")
        for i in range(MAX_ROUND):
            # print a_state
            a_action = rl.choose_action(a_state, b_state, episode <= TEST_MAX_EPISODES)
            a_nextstate, reward, a_action, result, b_nextstate = get_env_feedback("A", a_state, b_state, a_action,
                                                                                  [])


            if (result == "error"):
                print("a_action error")
                break
            if (episode > TEST_MAX_EPISODES and ShowGrapie):
                env.step("A", a_nextstate)
                env.render()
                time.sleep(FRESH_TIME)
            if (result == "gameover"):
                rl.store_transition(a_state, b_state, a_nextstate, b_nextstate, a_action, reward)
                win = "B"
                if (reward > 0):
                    if (episode > TEST_MAX_EPISODES and ShowGrapie):
                        env.step("B", b_nextstate)
                        env.render()
                        time.sleep(FRESH_TIME)
                    win = "A"
                step_counter += 1
                break

            b_action = get_b_action(a_nextstate, b_state, [])
            if (b_state == 22):
                print("b_state=" + str(b_state) + "; b_action=" + b_action)
            b_nextstate, reward, b_action, result, a_nextstate = get_env_feedback("B", b_state, a_nextstate, b_action,
                                                                                  [])

            if (result == "error"):
                print("b_action error")
                break
            if (episode > TEST_MAX_EPISODES and ShowGrapie):
                env.step("B", b_nextstate)
                env.render()
                time.sleep(FRESH_TIME)
            if (result == "gameover"):
                rl.store_transition(a_state, b_state, a_nextstate, b_nextstate, a_action, reward)
                win = "A"
                if (reward > 0):
                    if (episode > TEST_MAX_EPISODES and ShowGrapie):
                        env.step("A", a_nextstate)
                        env.render()
                        time.sleep(FRESH_TIME)
                    win = "B"
                step_counter += 1
                break
            #speichert die State, Actionen und Belohnungen für Trainierung
            rl.store_transition(a_state, b_state, a_nextstate, b_nextstate, a_action, reward)
            if (episode >= start_episode and episode % interval == 0):
                rl.lern()
            a_state = a_nextstate
            b_state = b_nextstate
            step_counter += 1
        if (win != "" and episode > TEST_MAX_EPISODES and ShowGrapie):
            env.reset()
            sum[win] += 1
        print("episode=" + str(episode) + "; step_counter=" + str(step_counter) + "; win=" + win)

    return sum


Type = "RL_DuelingDQN" #die Art des Trainierungsnetz
start_episode = 500 # die Anfangrunde der Trainierung
interval = 50 # das Interval der Trainierung
if __name__ == '__main__':
    allstates = create_all_state()
    env = Game(Actions)
    if (Type == "RL_DQN_1n"):
        from RL_DQN_1n import DQN

        start_episode = 200
        interval = 200
        rl = DQN(Actions)
    elif (Type == "RL_DQN"):
        from RL_DQN import DQN

        rl = DQN(Actions, memory_size=500, lr=0.01, batch_size=200, replace_target_iter=100)
    elif (Type == "RL_Double_DQN"):
        from RL_Double_DQN import DQN

        rl = DQN(Actions, memory_size=500, lr=0.01, batch_size=200, replace_target_iter=100)
    elif (Type == "RL_DuelingDQN"):
        from RL_DuelingDQN import DQN

        rl = DQN(Actions, memory_size=500, lr=0.01, batch_size=200, replace_target_iter=100)
    elif (Type == "RL_DQNPrioritizedReplay"):
        from RL_DQNPrioritizedReplay import DQN

        rl = DQN(Actions, memory_size=500, lr=0.01, batch_size=200, replace_target_iter=100)
    sum = update(env, rl, start_episode, interval)
    print(sum)
    rl.plot_cost()
    create_actions_table(rl)
