'''''
baut eine graphische Umgebung mit Tkinter für Präsentation. Das Schachbrett besteht hat 3 Zeilen und 3 Spalten.
insgesamt 9 Zellen. In der Zellen gibt es 3 schwarz Rechteck, die als Schwarzloch gelten. 
2 Spieler, A und B teilnehmen an dem Spil. A Spieler ist eine Torte in Rot und B Spieler in Gelb.
'''''
import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 60  # pixels der Zelle
Grid_H = 3  # Anzahl der Zeilen
Grid_W = 3  # Anzahl der Spalten


class Game(tk.Tk, object):
    def __init__(self, actions):
        super(Game, self).__init__()
        self.action_space = actions
        self.n_actions = len(self.action_space)
        self.title('spring loch')
        self.geometry('{0}x{1}'.format(Grid_H * UNIT, Grid_W * UNIT))
        # create origin
        self.origin = np.array([30, 30])
        self._build()

    #erstellt die Schachbrett und die 3 Löcher
    def _build(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=Grid_H * UNIT,
                                width=Grid_W * UNIT)

        # erstellt die Schachbrett
        for c in range(0, Grid_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, Grid_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, Grid_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, Grid_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # darstellt die Löcher
        center = self.origin + np.array([UNIT, UNIT * 2])
        self.loch = self.canvas.create_rectangle(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill='black')
        center = self.origin + np.array([UNIT * 2, 0])
        self.loch2 = self.canvas.create_rectangle(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill='black')
        center = self.origin + np.array([UNIT * 2, UNIT * 2])
        self.loch3 = self.canvas.create_rectangle(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill='black')
        self.A = False
        self.B = False
        self.canvas.pack()

    #setzt die Schachbrett zurück
    def reset(self):
        if (self.A):
            self.canvas.delete(self.A)

        if (self.B):
            self.canvas.delete(self.B)
        self.A = False
        self.B = False
        self.update()
        return

    #bewegt den Schachfigur
    def step(self, user, station):
        if (station != "gameover"):
            col = int(station / 10) - 1
            row = station % 10 - 1
            center = self.origin + np.array([UNIT * row, UNIT * col])

        if (user == "A"):
            if (self.A):
                if (station == "gameover"):
                    self.canvas.delete(self.A)
                    self.A = False
                else:
                    s = self.canvas.coords(self.A)
                    base_action = np.array([s[0] + 15, s[1] + 15])
                    base_action = center - base_action
                    self.canvas.move(self.A, base_action[0], base_action[1])  # move agent
            else:
                self.A = self.canvas.create_oval(center[0] - 15, center[1] - 15, center[0] + 15, center[1] + 15,
                                                 fill="red")
                self.canvas.coords(self.A)
        else:
            if (self.B):
                if (station == "gameover"):
                    self.canvas.delete(self.B)
                    self.B = False
                else:
                    s = self.canvas.coords(self.B)
                    base_action = np.array([s[0] + 15, s[1] + 15])
                    base_action = center - base_action
                    self.canvas.move(self.B, base_action[0], base_action[1])  # move agent
            else:
                self.B = self.canvas.create_oval(center[0] - 15, center[1] - 15, center[0] + 15, center[1] + 15,
                                                 fill="yellow")
                self.canvas.coords(self.B)

    #aktualisiert die graphische Darstellung
    def render(self):
        time.sleep(0.1)
        self.update()

#testet
if __name__ == '__main__':
    Actions = ["up", "right", "down", "left"]
    States = [11, 12, 13, 21, 22, 23, 31, 33]
    env = Game(Actions)
    time.sleep(1)

    env.step("A", 11)
    env.render()
    time.sleep(3)

    env.step("B", 21)
    env.render()
    time.sleep(5)

    env.step("A", 12)
    env.render()
    time.sleep(3)

    env.step("B", 22)
    env.render()
    time.sleep(5)

    env.step("A", 13)
    env.render()
    time.sleep(3)

    env.step("B", 23)
    env.render()
    time.sleep(5)

    env.step("A", 12)
    env.render()
    time.sleep(3)

    env.step("B", 22)
    env.render()
    time.sleep(5)
    env.reset()
