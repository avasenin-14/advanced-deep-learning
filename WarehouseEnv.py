import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WarehouseEnv(gym.Env):
    def __init__(
        self,
        grid_map,
        embeddings_dict,
        reward_config=None,
        max_steps=50,
        start_pos=(0, 0),
        random_start=False,
    ):
        super(WarehouseEnv, self).__init__()

        # grid_map IDs aligned with doc: 0=Floor, 1=Wall, 2=Pallet, 3=Sign
        self.grid_map = np.array(grid_map)
        if self.grid_map.ndim != 2:
            raise ValueError("grid_map must be a 2D array.")

        if self.grid_map.shape[0] != self.grid_map.shape[1]:
            raise ValueError("grid_map must be square (e.g., 5x5).")

        self.grid_size = self.grid_map.shape[0]

        # Find the pallet location dynamically from the map
        # This aligns with the "collect pallets" task in the doc [cite: 71]
        # argwhere returns [row, col] = [y, x], but agent_pos uses [x, y]
        target_positions = np.argwhere(self.grid_map == 2)
        if len(target_positions) == 0:
            raise ValueError("grid_map must include at least one pallet tile (ID 2).")
        self.target_coords = np.array([target_positions[0][1], target_positions[0][0]], dtype=np.int64)

        self.embeddings_dict = {
            int(k): np.asarray(v, dtype=np.float32) for k, v in embeddings_dict.items()
        }
        required_ids = set(np.unique(self.grid_map).tolist())
        available_ids = set(self.embeddings_dict.keys())
        if not required_ids.issubset(available_ids):
            missing = sorted(required_ids - available_ids)
            raise ValueError(f"embeddings_dict is missing keys for grid IDs: {missing}")

        self.embed_dim = int(next(iter(self.embeddings_dict.values())).shape[0])
        for obj_id, emb in self.embeddings_dict.items():
            if emb.ndim != 1 or emb.shape[0] != self.embed_dim:
                raise ValueError(
                    f"Embedding for ID {obj_id} must have shape ({self.embed_dim},), got {emb.shape}."
                )

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 + self.embed_dim,),
            dtype=np.float32
        )

        self.default_start_pos = np.array(start_pos, dtype=np.int64)
        if self.default_start_pos.shape != (2,):
            raise ValueError("start_pos must be a 2D coordinate in [x, y] format.")
        if not self._is_valid_cell(self.default_start_pos):
            raise ValueError("start_pos is out of bounds.")
        if self.grid_map[self.default_start_pos[1], self.default_start_pos[0]] == 1:
            raise ValueError("start_pos cannot be a wall tile (ID 1).")

        self.random_start = bool(random_start)
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.step_count = 0

        base_reward = {
            "step_penalty": -0.01,
            "wall_penalty": -0.20,
            "goal_reward": 10.0,
            "progress_bonus": 0.05,
        }
        if reward_config is not None:
            base_reward.update(reward_config)
        self.reward_config = base_reward

        self.agent_pos = self.default_start_pos.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        options = options or {}
        requested_start = options.get("start_pos")
        use_random_start = bool(options.get("random_start", self.random_start))

        if requested_start is not None:
            start = np.array(requested_start, dtype=np.int64)
            if start.shape != (2,):
                raise ValueError("options['start_pos'] must be [x, y].")
            if not self._is_valid_cell(start):
                raise ValueError("options['start_pos'] is out of bounds.")
            if self.grid_map[start[1], start[0]] == 1:
                raise ValueError("options['start_pos'] cannot be a wall tile.")
            self.agent_pos = start
        elif use_random_start:
            self.agent_pos = self._sample_random_start()
        else:
            self.agent_pos = self.default_start_pos.copy()

        obs = self._get_obs()
        info = {
            "distance_to_target": self._manhattan_distance(self.agent_pos, self.target_coords)
        }
        return obs, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected one of [0, 1, 2, 3].")

        prev_pos = self.agent_pos.copy()
        hit_wall = False

        new_pos = self.agent_pos.copy()
        if action == 0:   new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)  # Up
        elif action == 1: new_pos[1] = max(0, new_pos[1] - 1)                 # Down
        elif action == 2: new_pos[0] = max(0, new_pos[0] - 1)                 # Left
        elif action == 3: new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1) # Right

        # Check for Wall (ID 1) as defined in Part 2 [cite: 51]
        if self.grid_map[new_pos[1], new_pos[0]] == 1:
            hit_wall = True
        else:
            self.agent_pos = new_pos

        self.step_count += 1

        # Mission: Reach the Pallet (ID 2)
        terminated = np.array_equal(self.agent_pos, self.target_coords)
        truncated = (
            self.max_steps is not None
            and self.step_count >= self.max_steps
            and not terminated
        )

        reward = self._calculate_reward(prev_pos, self.agent_pos, terminated, hit_wall)

        info = {
            "hit_wall": hit_wall,
            "distance_to_target": self._manhattan_distance(self.agent_pos, self.target_coords),
            "step_count": self.step_count,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obj_id = self.grid_map[self.agent_pos[1], self.agent_pos[0]]
        embedding = self.embeddings_dict[obj_id]

        return np.concatenate([
            self.agent_pos.astype(np.float32),
            embedding.astype(np.float32)
        ])

    def _calculate_reward(self, prev_pos, current_pos, terminated, hit_wall):
        if terminated:
            return float(self.reward_config["goal_reward"])

        reward = float(self.reward_config["step_penalty"])

        if hit_wall:
            reward += float(self.reward_config["wall_penalty"])

        prev_dist = self._manhattan_distance(prev_pos, self.target_coords)
        curr_dist = self._manhattan_distance(current_pos, self.target_coords)
        reward += float(self.reward_config["progress_bonus"]) * float(prev_dist - curr_dist)

        return float(reward)

    def _is_valid_cell(self, pos):
        return (
            0 <= int(pos[0]) < self.grid_size
            and 0 <= int(pos[1]) < self.grid_size
        )

    @staticmethod
    def _manhattan_distance(pos_a, pos_b):
        return int(np.abs(pos_a[0] - pos_b[0]) + np.abs(pos_a[1] - pos_b[1]))

    def _sample_random_start(self):
        valid_cells = np.argwhere(self.grid_map != 1)
        if len(valid_cells) == 0:
            raise ValueError("No valid start cells found (all cells are walls).")

        # Keep random starts meaningful by avoiding the goal tile itself.
        non_goal_cells = [
            cell for cell in valid_cells if not np.array_equal(cell, np.array([self.target_coords[1], self.target_coords[0]]))
        ]
        candidates = non_goal_cells if len(non_goal_cells) > 0 else valid_cells

        idx = self.np_random.integers(0, len(candidates))
        y, x = candidates[idx]
        return np.array([x, y], dtype=np.int64)