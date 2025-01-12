# Konstanten
MAX_MEMORY = 100_000  # Maximale Anzahl von gespeicherten Erfahrungen.(auf vergangene Erfahrungen zurückgreifen)
BATCH_SIZE = 1000  # Anzahl der Erfahrungen, die in einem Schritt zum Training verwendet werden.
LR = 0.001  # Lernrate: Bestimmt, wie stark das Modell bei jedem Schritt angepasst wird.


# Definition der Klasse für das Q-Learning-Modell
class Linear_QNet(nn.Module):  #Ein neuronales Netz der geschätzten Q-Werte.

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # Erste lineare Schicht
        self.linear2 = nn.Linear(hidden_size, output_size)  # Zweite lineare Schicht

    def forward(self, x):
        x = F.relu(self.linear1(x))  # Aktivierung der ersten Schicht mit ReLU (Nichtlineare Bz lernen:komplexe Beziehungen)
        x = self.linear2(x)          # Ausgabe durch die zweite Schicht
        return x                     

    def save(self, file_name='model.pth'):
        model_folder_path = './model'  # Ordner für das Speichern der Modelle
        if not os.path.exists(model_folder_path):  
            os.makedirs(model_folder_path)        

        file_name = os.path.join(model_folder_path, file_name) 
        torch.save(self.state_dict(), file_name)               # Speichern der Modellparameter

# Definition der Trainingslogik für Q-Learning (Bestandteil RL)
class QTrainer: 
    def __init__(self, model, lr, gamma):
        self.lr = lr  
        self.gamma = gamma                     
        self.model = model                     
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # Adam-Optimierer (verbessert klassischen Gradientenabstiegsalgorithmus)
        self.criterion = nn.MSELoss()          # Verlustfunktion: Mean Squared Error(Vorhersagewert optimieren)

    def train_step(self, state, action, reward, next_state, done):
        # Konvertieren der Eingaben in Tensoren (NN Erforderlich)
        state = torch.tensor(state, dtype=torch.float32)       # Zustand
        next_state = torch.tensor(next_state, dtype=torch.float32)  
        action = torch.tensor(action, dtype=torch.long)        # Aktion
        reward = torch.tensor(reward, dtype=torch.float32)     # Belohnung

        
        if len(state.shape) == 1:              # Falls die Eingabe eindimensional ist
            state = torch.unsqueeze(state, 0)  # Hinzufügen einer Batch-Dimension
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )                    # Umwandeln von done in ein Tupel

        # Vorhersage der aktuellen Q-Werte
        pred = self.model(state)               

        # Berechnung der Ziel-Q-Werte
        target = pred.clone()                  # Kopieren der vorhergesagten Q-Werte
        for idx in range(len(done)):           # Iteration über alle Einträge im Batch
            Q_new = reward[idx]                # Initialer Q-Wert ist die Belohnung
            if not done[idx]:                  # Wenn nicht abgeschlossen 
                with torch.no_grad():          # Keine Gradientenberechnung
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))  # Aktualisieren

            target[idx][torch.argmax(action[idx]).item()] = Q_new  # Ziel-Q-Wert setzen

        # Berechnung des Verlusts und Backpropagation
        self.optimizer.zero_grad()             # Zurücksetzen der Gradienten
        loss = self.criterion(target, pred)    # Verlust berechnen
        loss.backward()                        # Backpropagation durchführen
        self.optimizer.step()                  