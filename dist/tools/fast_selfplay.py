import subprocess
import multiprocessing as mp
import os
import sys
import re
from functools import partial

# Configuration

AGENT_A_NAME = "AgentEProMax"
AGENT_B_NAME = "AgentEProMaxPlus"

GAMES = 24             # total number of games to run
PARALLEL = 8           # number of processes to run simultaneously
SILENT = True          # hide engine logs (must be True to parse output effectively)


def run_single_game(i, agentA_name, agentB_name, silent):
    """
    Runs exactly ONE game and returns:
    (winner_code, score_a, score_b, final_log, trap_events_str, trap_count, p1_agent, p2_agent)
    """

    # Alternate who is Player 1
    if i % 2 == 1:
        p1_agent = agentA_name
        p2_agent = agentB_name
    else:
        p1_agent = agentB_name
        p2_agent = agentA_name
    
    cmd = [
        sys.executable,
        "engine/run_local_agents.py",
        p1_agent,
        p2_agent,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout
    except Exception as e:
        return 0, 0, 0, f"CRASH: {e}", "", 0, p1_agent, p2_agent

    # ---------------------------
    # 1. Winner
    # ---------------------------
    winner_code = 0
    if "PLAYER_A wins" in output:
        winner_code = 1
    elif "PLAYER_B wins" in output:
        winner_code = -1

    # ---------------------------
    # 2. Scores
    # ---------------------------
    matches = re.findall(r"EGGS\s+A:(\d+)\s+B:(\d+)", output)
    score_p1 = 0
    score_p2 = 0
    if matches:
        score_p1 = int(matches[-1][0])
        score_p2 = int(matches[-1][1])

    # ---------------------------
    # 3. Trap Event Parsing  (Improved)
    # ---------------------------
    trap_pattern = (
        r"Triggered trapdoor at \((\d+),\s*(\d+)\),\s*([AB]) returned to "
        r"\((\d+),\s*(\d+)\)"
    )
    events = re.findall(trap_pattern, output)

    trap_events = []
    for (tx, ty, who, rx, ry) in events:
        agent_name = p1_agent if who == "A" else p2_agent
        trap_events.append(
            f"{agent_name} triggered trapdoor at ({tx},{ty}), returned to ({rx},{ry})"
        )

    trap_events_str = "\n".join(trap_events)
    trap_count = len(events)

    # ---------------------------
    # 4. Trim log for readability
    # ---------------------------
    lines = output.strip().split("\n")
    final_log = "\n".join(lines[-40:])  # last 40 lines

    return (
        winner_code,
        score_p1,
        score_p2,
        final_log,
        trap_events_str,
        trap_count,
        p1_agent,
        p2_agent,
    )


def main():
    print(f"Running {GAMES} games ({AGENT_A_NAME} vs {AGENT_B_NAME}) with {PARALLEL} workers...\n")
    pool = mp.Pool(PARALLEL)

    worker = partial(run_single_game,
                     agentA_name=AGENT_A_NAME,
                     agentB_name=AGENT_B_NAME,
                     silent=SILENT)

    a_wins = 0
    b_wins = 0
    draws = 0
    total_score_a = 0
    total_score_b = 0

    print(f"{'Game':<6} | {'Winner':<20} | {'Score A':<8} | {'Score B':<8} | {'Trap Hits':<12} | {'Winrate A'}")
    print("-" * 90)

    for idx, result in enumerate(pool.imap_unordered(worker, range(GAMES)), 1):
        (
            winner_code,
            score_p1,
            score_p2,
            final_log,
            trap_events,
            trap_count,
            p1_agent,
            p2_agent,
        ) = result

        # Map scores to Agent A / Agent B
        is_agentA_p1 = (
            (idx % 2 == 1 and p1_agent == AGENT_A_NAME)
            or (idx % 2 == 0 and p2_agent == AGENT_A_NAME)
        )

        if is_agentA_p1:
            score_A_final = score_p1
            score_B_final = score_p2
        else:
            score_A_final = score_p2
            score_B_final = score_p1

        # Winner name
        winner_name = "Draw"
        if winner_code == 1:
            winner_name = p1_agent
            if winner_name == AGENT_A_NAME: a_wins += 1
            else: b_wins += 1
        elif winner_code == -1:
            winner_name = p2_agent
            if winner_name == AGENT_A_NAME: a_wins += 1
            else: b_wins += 1
        else:
            draws += 1

        total_score_a += score_A_final
        total_score_b += score_B_final

        trap_display = f"({trap_count})" if trap_count else "-"

        current_winrate_a = a_wins / idx

        # Header line
        print(
            f"#{idx:<5} | {winner_name:<20} | {score_A_final:<8} "
            f"| {score_B_final:<8} | {trap_display:<12} | {current_winrate_a:.2%}"
        )
        print("-" * 90)

        # Trap event details
        if trap_events:
            print("    🚨 TRAP EVENTS:")
            for line in trap_events.split("\n"):
                print(f"    {line}")
            print("-" * 90)

        # Final board state
        for line in final_log.split("\n"):
            print(f"    {line}")
        print("-" * 90)

    pool.close()
    pool.join()

    print("\n=== FINAL RESULTS (Alternating First Move) ===")
    print(f"Total Games: {GAMES}")
    print(f"{AGENT_A_NAME} Wins: {a_wins}")
    print(f"{AGENT_B_NAME} Wins: {b_wins}")
    print(f"Draws:       {draws}")
    print("-" * 30)
    print(f"{AGENT_A_NAME} Winrate:   {a_wins / GAMES:.2%}")
    print(f"Avg Score {AGENT_A_NAME}: {total_score_a / GAMES:.2f}")
    print(f"Avg Score {AGENT_B_NAME}: {total_score_b / GAMES:.2f}")
    print(f"Avg Diff ({AGENT_A_NAME} - {AGENT_B_NAME}): {(total_score_a - total_score_b) / GAMES:+.2f}")


if __name__ == "__main__":
    main()
