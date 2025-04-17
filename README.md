# genetic-tic-tac-toe
A Tic-Tac-Toe game powered by a Genetic Algorithm that evolves intelligent strategies over generations. This project demonstrates how evolutionary computation can be applied to game strategy optimization.
# Unbeatable Tic-Tac-Toe AI with Genetic Algorithm

An AI-powered Tic-Tac-Toe game where the computer uses a genetic algorithm to learn optimal strategies. The AI trains locally and becomes unbeatable over generations.

![Game Screenshot](screenshot.png) *(Replace with your actual screenshot later)*

## Features
- 🧠 **Self-learning AI**: Trains using genetic algorithms (no pre-trained models)
- ⚡ **100% Offline**: No internet required after initial setup
- 🏆 **Unbeatable**: AI maximizes win-rate through evolution
- 🎮 **Clean GUI**: Built with Python's Tkinter

## How It Works
1. **Genetic Algorithm** evolves AI strategies by:
   - Evaluating board states with heuristic weights
   - Selecting top-performing "chromosomes"
   - Crossbreeding and mutating strategies
2. **Heuristic Features**:
3. 
   - Immediate win opportunities
   - Blocking opponent wins
   - Center/corner control

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Adi-112/genetic-tic-tac-toe
   cd tic-tac-toe-ai

2. Install dependencies:
   pip install numpy
   python tic_tac_toe_ai.py
   First run will train the AI (~2-5 minutes)

Subsequent runs use the trained best_ai.pkl

.
├── tic_tac_toe_ai.py      # Main game + AI code
├── best_ai.pkl            # Trained AI weights (auto-generated)
├── README.md              # This file
└── requirements.txt       # Dependencies
3. Customization

Adjust difficulty: Modify train_ai() parameters:

train_ai(generations=50, pop_size=100)  # Increase for stronger AI
Change heuristics: Edit calculate_heuristic() in code
4.Contributing
Pull requests welcome! For major changes, open an issue first.

License


Copy

---

### **Additional GitHub Setup Steps**

1. **Create a repository** on GitHub (don't initialize with README)
2. **Upload files**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/Adi-112/genetic-tic-tac-toe
   git push -u origin main

