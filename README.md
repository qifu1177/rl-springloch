# rl-springloch
ein Beispiel mit Reinforcement Learning.
1. die Beschreibung des Spiel

Der Beispiel ist ein Schachspiel. Auf dem Schachbrett gibt es 3 Zeilen, 3 Spalten, insgesamt 9 Zellen und darin gibt es noch 3 Löcher.
Zwei Spieler sind KI und Computer, der den Schachfigur nur zufällig bewegen kann. Jede Spieler hat nur einen Schachfigur.
Wessen Schachfigur geht  auf der Plätze der Löcher, wer verliert.

2. die Dateiliste
    1) game_env.py ist eine graphische Darstellung und präsentiert den Spielvorgang.
    2) Rl_brain.py ist ein Q-learning Programm
    3) run_this.py ist das haupte Programm, in dem die Dateien "game_env.py" und "RL_brain.py" importiert werden. Es erstellt die Laufenumgebung, definiert die Spielregel, entscheidet, wer gewinnt, und lernt.
    4) RL_DQN_1n.py ist ein Deep Q-learning Programm, das einen Neuronales Netz, das zwei Schicht hat, eingesetzt.
    5) RL_DQN.py ist ein Deep Q-learning Programm, das zwei Neuronales Netz hat, eine die aktuelle State evaluiert, die loss-Werte berechnet und die Parameter optimiert;
    die Andere die nächste State evaluiert, keine Optimierung der Parameter hat und deren Parameter aus der Netz 1 pro n(z.B 100 oder mehr) Schritte aktuallisiert werden.
    6) RL_Double_DQN ist eine Variante von RL_DQN.Das Ziel ist so, die Konvergenz zu verbessern. die Unterschiede ist bei der Berechnung der Q_Werte für die nächste State.
    7) RL_DuelingDQN ist eine Variante von RL_DQN. Die Netz berechnet zwei Q_Werte, eine(V) nur eine Spalte hat und die Andere(A) wieviele Spalte wie Actions ist, hat; und dann die letzte Q-Wert(q_eval) durch den fogende Formel brechnet wird.
    Q = V(s) + A(s,a)(code: q_eval = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True)) )
    8) RL_DQNPrioritizedReplay ist andere Variante von RL_DQN. Die kann die wichtige Schritte, die nach der Differenz zwischen Q_target und Q_eval sortiert werden, mehr lernen.
    9) run_dqn.py ist das haupte Programm für alle DQN, das geleiche Funktionen wie run_this. Die kann das unterschiede DQN-Programm durch Änderung des Attribut Type auswahlen.

3. Installation

  man muss pandas, numpy,matplotlib, tensorflow und tkinter installieren.
  Für pandas, numpy und matplotlib kann man durch den Befehl "pip3 install lib-name" installieren.
  Installation der Tensorflow, dass man die Seite "https://www.tensorflow.org/install/" durchlesen kann.
  tkinter hat keine Windows-Version, Linux Nutzer führt den Befehl "apt-get install python3-tk" aus und bei Macos ist "brew       install python3-tk".
  
4. Ausführung
  1) öffnen die Terminal und zu Ordner "rl-springloch"
  2) testen Q_learnig, dann geben den Befehl "python3 run_this.py"
  3) testen Deep Q_learning, dann geben den Befehl "python3 run_dqn.py".

