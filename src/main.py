# src/main.py

import argparse
import os

import torch
from data_loader import load_words, generate_training_data, WordleDataset, generate_word_files
from model import HeuristicScoringModel
from train import train_model
from gameplay import play_wordle
from torch.utils.data import DataLoader
from config import FAST_MODE
from torch.cuda import amp
from tqdm import tqdm

def main():
    scaler = amp.GradScaler()
    parser = argparse.ArgumentParser(description="Wordle Solver using ML")
    parser.add_argument('--mode', type=str, choices=['train', 'play', 'test'], default='train', help='Mode to run the script in.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--game', type=str, help='Target word for gameplay.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")  # Essential output kept outside FAST_MODE

    # Removed time.sleep(5) for faster startup

    print("[INFO] Ensuring word files are generated...")  # Essential output
    generate_word_files('../data/words.txt', '../data/allowed_guesses.txt', '../data/possible_solutions.txt')

    print("[INFO] Loading allowed guesses and possible solutions...")  # Essential output
    allowed_guesses = load_words('../data/allowed_guesses.txt')
    possible_solutions = load_words('../data/possible_solutions.txt')

    input_size = 312

    if args.mode == 'train':
        print("[INFO] Generating training data...")  # Essential output
        training_data = generate_training_data(possible_solutions, allowed_guesses, num_games=10000)
        print(f"[INFO] Training data generated: {len(training_data)} samples")  # Essential output

        dataset = WordleDataset(training_data)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

        model = HeuristicScoringModel(input_size).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print("[INFO] Starting training...")  # Essential output
        train_model(model, dataloader, criterion, optimizer, device, scaler, args.epochs)

        torch.save(model.state_dict(), 'model.pth')
        print("[INFO] Training completed and model saved as 'model.pth'.")  # Essential output

    elif args.mode == 'play':
        if not args.game:
            print("[ERROR] Please provide a target word using --game")
            return
        target_word = args.game.upper()
        if len(target_word) != 5 or not target_word.isalpha():
            print("[ERROR] Please provide a valid 5-letter target word.")
            return
        if target_word not in possible_solutions:
            print("[ERROR] Target word not in the possible solutions list.")
            return

        model = HeuristicScoringModel(input_size).to(device)
        try:
            model.load_state_dict(torch.load('model.pth', map_location=device))
            model.eval()
        except FileNotFoundError:
            print("[ERROR] Model file 'model.pth' not found. Please train the model first using --mode train.")
            return

        print(f"[INFO] Starting Wordle game with target: {target_word}")  # Essential output
        attempts_taken = play_wordle(model, target_word, allowed_guesses, possible_solutions, device)
        print(f"[INFO] Word '{target_word}' solved in {attempts_taken} attempts.")  # Essential output

    elif args.mode == 'test':
        # **Testing Mode Implementation**
        if not os.path.exists('model.pth'):
            print("[ERROR] Model file 'model.pth' not found. Please train the model first using --mode train.")
            return

        model = HeuristicScoringModel(input_size).to(device)
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model.eval()
        print("[INFO] Loaded trained model for testing.")

        total_attempts = 0
        total_solved = 0
        total_words = len(possible_solutions)

        print("[INFO] Starting testing over all possible solutions...")
        for target_word in tqdm(possible_solutions, desc="Testing Progress"):
            attempts = play_wordle(model, target_word, allowed_guesses, possible_solutions, device, verbose=False)
            if attempts <= 6:  # Assuming 6 attempts are allowed
                total_solved += 1
                total_attempts += attempts
            else:
                # If the solver fails to guess within allowed attempts
                total_attempts += 6  # Count as maximum attempts

        average_guesses = total_attempts / total_words
        accuracy = (total_solved / total_words) * 100

        print("\n[RESULTS]")
        print(f"Total Words Tested: {total_words}")
        print(f"Words Solved: {total_solved}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average Number of Guesses: {average_guesses:.2f}")


if __name__ == "__main__":
    main()
