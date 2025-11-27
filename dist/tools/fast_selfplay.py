import subprocess
import multiprocessing as mp
import os
import sys
import re
from functools import partial

# Configuration

AGENT_A = "AgentD"
AGENT_B = "AgentE"

GAMES = 24             # total number of games to run
PARALLEL = 8           # number of processes to run simultaneously
SILENT = True          # hide engine logs (must be True to parse output effectively)


def run_single_game(i, agentA, agentB, silent):
    """
    Runs exactly ONE game and returns a tuple:
    (winner_code, score_a, score_b, final_log)
    """

    cmd = [
        sys.executable,
        "engine/run_local_agents.py",
        agentA,
        agentB,
    ]

    try:
        # Capture stdout
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout
    except Exception as e:
        return 0, 0, 0, f"CRASH: {e}"

    # 1. Detect Winner
    winner = 0
    if "PLAYER_A wins" in output:
        winner = 1
    elif "PLAYER_B wins" in output:
        winner = -1

    # 2. Parse Scores
    matches = re.findall(r"EGGS\s+A:(\d+)\s+B:(\d+)", output)
    score_a = 0
    score_b = 0
    if matches:
        final_score = matches[-1]
        score_a = int(final_score[0])
        score_b = int(final_score[1])

    # 3. Extract Final Game State (Board + Result)
    # We grab the last ~20 lines which typically contain the final turn board and win message
    lines = output.strip().split('\n')
    final_log = "\n".join(lines[-40:])

    return winner, score_a, score_b, final_log


# Parallel execution

def main():
    print(f"Running {GAMES} games ({AGENT_A} vs {AGENT_B}) with {PARALLEL} workers...\n")
    pool = mp.Pool(PARALLEL)

    worker = partial(run_single_game,
                     agentA=AGENT_A,
                     agentB=AGENT_B,
                     silent=SILENT)

    # Stats tracking
    a_wins = 0
    b_wins = 0
    draws = 0
    total_score_a = 0
    total_score_b = 0

    print(f"{'Game':<6} | {'Winner':<8} | {'Score':<10} | {'Cumulative WinRate'}")
    print("-" * 55)

    for idx, (winner, s_a, s_b, final_log) in enumerate(pool.imap_unordered(worker, range(GAMES)), 1):
        # Update counts
        winner_str = "Draw"
        if winner == 1: 
            a_wins += 1
            winner_str = "A"
        elif winner == -1: 
            b_wins += 1
            winner_str = "B"
        else: 
            draws += 1
        
        total_score_a += s_a
        total_score_b += s_b

        # Print Result Header
        print(f"#{idx:<5} | {winner_str:<8} | {s_a}-{s_b:<7} | {a_wins/idx:.1%}")
        
        # Print Final Board State (Indented for readability)
        print("-" * 30)
        for line in final_log.split('\n'):
            print(f"    {line}")
        print("-" * 55)

    print("\n=== FINAL RESULTS ===")
    print(f"Total Games: {GAMES}")
    print(f"A Wins:      {a_wins}")
    print(f"B Wins:      {b_wins}")
    print(f"Draws:       {draws}")
    print("-" * 20)
    print(f"A Winrate:   {a_wins / GAMES:.2%}")
    print(f"Avg Score A: {total_score_a / GAMES:.2f}")
    print(f"Avg Score B: {total_score_b / GAMES:.2f}")
    print(f"Avg Diff:    {(total_score_a - total_score_b) / GAMES:+.2f}")

if __name__ == "__main__":
    main()