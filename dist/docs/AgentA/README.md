# AgentA - Strategic Chicken Game Agent

AgentA is a strategic agent designed to win the chicken game by maximizing eggs laid while avoiding trapdoors and strategically using resources.

## Strategy Overview

AgentA implements the following strategies:

1. **Maximize Eggs**: Prioritizes laying eggs, especially at corners (worth 3 points vs 1)
2. **Trapdoor Avoidance**: Tracks trapdoor locations using sensor data (hear/feel) to avoid stepping on them
3. **Strategic Blocking**: Uses turds to block enemy when ahead in eggs or when enemy is nearby
4. **End Game Awareness**: Increases urgency for egg-laying moves when near the 40-move limit
5. **Parity Awareness**: Only attempts to lay eggs on squares matching the chicken's parity

## How to Run Locally

### Prerequisites

- Python 3.x
- The game engine files in `dist/engine/`
- Agent files in `dist/docs/AgentA/`

### Running a Game

The game can be run using the `run_local_agents.py` script. However, you need to set up the directory structure correctly.

#### Option 1: Using the existing structure

1. Navigate to the `dist/engine/` directory:
   ```bash
   cd dist/engine
   ```

2. Run the game with two agents:
   ```bash
   python run_local_agents.py AgentA Yolanda
   ```

   This will run AgentA vs Yolanda. You can replace `Yolanda` with any other agent name.

#### Option 2: Setting up a proper agents directory

The `run_local_agents.py` script expects agents to be in a `3600-agents` directory at the project root. You can either:

1. Create a symlink or copy:
   ```bash
   # From project root
   mkdir -p 3600-agents
   cp -r dist/docs/AgentA 3600-agents/
   cp -r dist/docs/Yolanda 3600-agents/  # if you want to test against Yolanda
   ```

2. Or modify `run_local_agents.py` to point to `dist/docs` instead of `3600-agents`.

#### Option 3: Direct Python execution (for testing)

If you want to test the agent directly, you can create a simple test script:

```python
import sys
import os

# Add the engine directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../engine'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.game_map import GameMap
from game.board import Board
from game.trapdoor_manager import TrapdoorManager
from AgentA.agent import PlayerAgent

# Create a test game
game_map = GameMap()
trapdoor_manager = TrapdoorManager(game_map)
board = Board(game_map, time_to_play=360, build_history=False)

# Initialize spawns and trapdoors
spawns = trapdoor_manager.choose_spawns()
trapdoor_locations = trapdoor_manager.choose_trapdoors()
board.chicken_player.start(spawns[0], 0)
board.chicken_enemy.start(spawns[1], 1)

# Create agent
def time_left():
    return 360.0

agent = PlayerAgent(board, time_left)

# Test a move
sensor_data = trapdoor_manager.sample_trapdoors(board.chicken_player.get_location())
move = agent.play(board, sensor_data, time_left)
print(f"Agent chose: {move}")
```

### Important Notes

1. **Path Setup**: The game engine adds the agent directory to `sys.path`, so agents can import from `game` module directly.

2. **Module Structure**: Agents must be in a directory with:
   - `__init__.py` file
   - `agent.py` file containing a `PlayerAgent` class

3. **Import Path**: Agents use `from game import *` which works because the engine adds the `dist/engine` directory to the Python path.

4. **Time Management**: The agent receives a `time_left()` callable that returns remaining time. AgentA makes quick decisions to avoid timeouts.

## Game Rules Considered

AgentA follows all rules from `ending_conditions.txt`:

1. ✅ **Invalid Moves**: Always uses `get_valid_moves()` to ensure moves are valid
2. ✅ **Time Management**: Makes quick decisions without unnecessary delays
3. ✅ **40 Move Limit**: Prioritizes eggs more when turns are running out
4. ✅ **Blocking**: Uses turds strategically to potentially block enemy (enemy gets bonus eggs if blocked)
5. ✅ **Trapdoor Avoidance**: Tracks and avoids trapdoors to prevent enemy from getting 4 eggs

## Key Features

- **Trapdoor Tracking**: Uses sensor data (hear/feel) to build a belief map of trapdoor locations
- **Corner Prioritization**: Gives extra weight to corner eggs (3x value)
- **Strategic Turd Placement**: Uses turds when ahead to maintain lead or when enemy is close
- **End Game Optimization**: Increases egg-laying priority in final 10 turns
- **Safety First**: Avoids suspected trapdoor locations unless no other options exist

## Testing

To test AgentA against Yolanda (random agent):

```bash
cd dist/engine
python run_local_agents.py AgentA Yolanda
```

The game will display the board and show the moves. Results are saved to `3600-agents/matches/` directory.

