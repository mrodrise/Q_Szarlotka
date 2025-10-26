from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
from gymnasium.envs.box2d import lunar_lander
import numpy as np
import pygame
import torch
from torch import nn


# Paths to the pretrained models that ship with the project.
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
QUANTUM_ACTOR_MODEL = MODELS_DIR / "Quantum_Actor_Model.pt"
QUANTUM_CRITIC_MODEL = MODELS_DIR / "Quantum_Critic_Model.pt"
TRADITIONAL_DQN_MODEL = MODELS_DIR / "Classic_DQN_Model.pt"
QUANTUM_DQN_MODEL = MODELS_DIR / "Quantum_DQN_Model.pt"

# Base configuration for the Lunar Lander matchup.
ENV_ID = "LunarLander-v3"
PANEL_WIDTH = 600
PANEL_HEIGHT = 480
FPS = 30
HUMAN_MAIN_THRUST_SCALE = 0.6
HUMAN_SIDE_THRUST_SCALE = 0.4


@dataclass
class EpisodeScore:
    current: float = 0.0
    last: Optional[float] = None
    best: Optional[float] = None

    def reset_active_episode(self) -> None:
        # Clear the running episode but keep the accumulated high score.
        self.current = 0.0
        self.last = None

    def add_reward(self, reward: float) -> None:
        # Incrementally track the reward stream coming from the environment.
        self.current += reward

    def finalize_episode(self) -> None:
        # Archive the final tally and update the best attempt if appropriate.
        self.last = self.current
        if self.best is None or self.current > self.best:
            self.best = self.current
        self.current = 0.0


class DQNNetwork(nn.Module):
    """Simple feed-forward network compatible with the provided checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        # Mirror the topology used during training so the checkpoints can load verbatim.
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



class QuantumDQNNetwork(nn.Module):
    """Quantum-inspired head used for legacy DQN checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.action_head = QuantumActionHead()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.action_head(state)
        return logits[..., :4]

class DQNAgent:
    """Loads a trained DQN and produces greedy actions."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")
        self.device = device
        state_dict = torch.load(checkpoint_path, map_location=device)
        if any(key.startswith("model.") for key in state_dict.keys()) and not any(key.startswith("model.encoder") for key in state_dict.keys()):
            self.network = DQNNetwork().to(self.device)
        elif "action_head.x_weights" in state_dict:
            self.network = QuantumDQNNetwork().to(self.device)
        elif any(key.startswith("model.encoder.quantum_layers") for key in state_dict):
            from DQN.config import DEFAULT_CONFIG
            from DQN.quantum_network import build_quantum_model
            settings = DEFAULT_CONFIG.get_environment("lunarlander")
            self.network = build_quantum_model(settings, DEFAULT_CONFIG.quantum_device).to(self.device)
            state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}

        else:
            raise RuntimeError(f"Unsupported DQN checkpoint format: {checkpoint_path}")
        self.network.load_state_dict(state_dict)
        self.network.eval()

    def act(self, state: np.ndarray) -> int:
        # Greedy policy: pick the action with the highest predicted Q-value.
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(tensor)
        return int(torch.argmax(q_values, dim=1).item())


class QuantumActionHead(nn.Module):
    """Lightweight layer storing the quantum-inspired actor parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.x_weights = nn.Parameter(torch.zeros(5, 8))
        self.z_weights = nn.Parameter(torch.zeros(5, 8))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Encode the input with paired sinusoidal rotations, mirroring training.
        x_rot = torch.matmul(state, self.x_weights.T)
        z_rot = torch.matmul(state, self.z_weights.T)
        return torch.sin(x_rot) + torch.cos(z_rot)


class ActorNetwork(nn.Module):
    """Actor head that turns environment observations into action logits."""

    def __init__(self) -> None:
        super().__init__()
        self.action_head = QuantumActionHead()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.action_head(state)
        return logits[..., :4]


class CriticNetwork(nn.Module):
    """Critic that evaluates state value estimates."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(8, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        value = self.linear3(x)
        return value.squeeze(-1)


class ActorCriticAgent:
    """Agent wrapper that loads frozen actor and critic checkpoints."""

    def __init__(
        self, actor_path: Path, critic_path: Path, device: torch.device
    ) -> None:
        if not actor_path.exists():
            raise FileNotFoundError(f"Actor model not found: {actor_path}")
        if not critic_path.exists():
            raise FileNotFoundError(f"Critic model not found: {critic_path}")

        self.device = device
        self.actor = ActorNetwork().to(device)
        self.critic = CriticNetwork().to(device)

        actor_state = torch.load(actor_path, map_location=device)
        critic_state = torch.load(critic_path, map_location=device)
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)

        self.actor.eval()
        self.critic.eval()

    def act(self, state: np.ndarray) -> int:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(tensor)
            probabilities = torch.softmax(logits, dim=-1)
        return int(torch.argmax(probabilities, dim=-1).item())

    def value(self, state: np.ndarray | torch.Tensor) -> float:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            estimate = self.critic(tensor)
        return float(estimate.item())


class GameApp:
    """Main application orchestrating the menu and the split-screen gameplay."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Lunar Lander: Human vs AI")

        self.screen = pygame.display.set_mode((PANEL_WIDTH * 2, PANEL_HEIGHT), pygame.RESIZABLE)
        # Cache fonts and timing utilities once to avoid churn during the loop.
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("arial", 20)
        self.font_medium = pygame.font.SysFont("arial", 28)
        self.font_large = pygame.font.SysFont("arial", 40)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.running = True
        self.state = "menu"

        self.ai_agent: Optional[object] = None
        self.ai_env: Optional[gym.Env] = None
        self.human_env: Optional[gym.Env] = None

        self.ai_state: Optional[np.ndarray] = None
        self.human_state: Optional[np.ndarray] = None
        self.ai_value_estimate: Optional[float] = None
        self.ai_score = EpisodeScore()
        self.human_score = EpisodeScore()

        self.info_message = ""
        self._update_layout()

    def run(self) -> None:
        # Main loop: route events to the current scene and redraw at a steady cadence.
        while self.running:
            events = pygame.event.get()
            if self.state == "menu":
                self._handle_menu(events)
            elif self.state == "controls":
                self._handle_controls(events)
            elif self.state == "game":
                self._handle_game(events)
            pygame.display.flip()
            self.clock.tick(FPS)

        self._cleanup_envs()
        pygame.quit()

    def _handle_menu(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self._handle_resize_event(event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self._start_game("dqn_neural")
                elif event.key == pygame.K_2:
                    self._start_game("dqn_quantum")
                elif event.key == pygame.K_3:
                    self._start_game("actor_critic")
                elif event.key == pygame.K_4:
                    self.state = "controls"
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

        self._draw_menu_screen()

    def _handle_controls(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN and event.key in {pygame.K_ESCAPE, pygame.K_BACKSPACE}:
                self.state = "menu"
                return
            if event.type == pygame.VIDEORESIZE:
                self._handle_resize_event(event.w, event.h)
        self._draw_controls_screen()

    def _handle_game(self, events: list[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.VIDEORESIZE:
                self._handle_resize_event(event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._return_to_menu()
                    return
                if event.key == pygame.K_r:
                    self._reset_episode()

        if not self.ai_env or not self.human_env or self.ai_state is None or self.human_state is None:
            return

        human_actions = self._determine_human_actions()
        self._step_human(human_actions)
        self._step_ai()
        self._draw_gameplay_screen()

    def _start_game(self, mode: str) -> None:
        try:
            if mode == "dqn_neural":
                self.ai_agent = DQNAgent(TRADITIONAL_DQN_MODEL, self.device)
            elif mode == "actor_critic":
                self.ai_agent = ActorCriticAgent(
                    QUANTUM_ACTOR_MODEL, QUANTUM_CRITIC_MODEL, self.device
                )
            elif mode == "dqn_quantum":
                self.ai_agent = DQNAgent(QUANTUM_DQN_MODEL, self.device)
            else:
                self.info_message = f"Unknown mode requested: {mode}"
                return
        except FileNotFoundError as exc:
            self.info_message = str(exc)
            return

        try:
            self._cleanup_envs()
            self.ai_env = gym.make(ENV_ID, render_mode="rgb_array")
            self.human_env = gym.make(ENV_ID, render_mode="rgb_array")
            self.ai_score = EpisodeScore()
            self.human_score = EpisodeScore()
            self._reset_episode()
            self.state = "game"
        except Exception as exc:  # noqa: BLE001
            self.info_message = f"Failed to initialise environments: {exc}"
            self.ai_agent = None

    def _reset_episode(self) -> None:
        if self.ai_env:
            self.ai_state, _ = self.ai_env.reset()
            self.ai_score.reset_active_episode()
        if self.human_env:
            self.human_state, _ = self.human_env.reset()
            self.human_score.reset_active_episode()
        self._update_ai_value_estimate()
        self.info_message = ""

    def _return_to_menu(self) -> None:
        self.state = "menu"
        self._cleanup_envs()
        self.ai_agent = None

    def _cleanup_envs(self) -> None:
        for env in (self.ai_env, self.human_env):
            if env is not None:
                env.close()
        self.ai_env = None
        self.human_env = None
        self.ai_state = None
        self.human_state = None
        self.ai_value_estimate = None

    def _handle_resize_event(self, width: int, height: int) -> None:
        # Force an even width so each side gets a whole pixel count.
        width = max(2, width)
        if width % 2:
            width -= 1
        height = max(1, height)
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self._update_layout()

    def _update_layout(self) -> None:
        # Keep track of the current viewport and how much space each side can use.
        width, height = self.screen.get_size()
        width = max(width, 2)
        height = max(height, 1)
        self.window_width = width
        self.window_height = height
        self.left_panel_width = width // 2
        self.right_panel_width = width - self.left_panel_width
        if self.left_panel_width == 0:
            self.left_panel_width = 1
        if self.right_panel_width == 0:
            self.right_panel_width = 1
        self.panel_height = height
        self.divider_x = self.left_panel_width

    def _determine_human_actions(self) -> list[int]:
        keys = pygame.key.get_pressed()
        # Lunar Lander action mapping:
        # 0: idle, 1: left thruster, 2: main thruster, 3: right thruster.
        # We allow stacking inputs so the player can feather multiple engines per frame.
        actions: list[int] = []
        if keys[pygame.K_LEFT]:
            actions.append(3)
        if keys[pygame.K_RIGHT]:
            actions.append(1)
        if keys[pygame.K_UP]:
            actions.append(2)
        if not actions:
            actions.append(0)
        return actions

    def _step_human(self, actions: list[int]) -> None:
        if not self.human_env or self.human_state is None:
            return
        with self._scaled_human_thrusters():
            for action in actions:
                next_state, reward, terminated, truncated, _ = self.human_env.step(action)
                self.human_score.add_reward(reward)
                self.human_state = next_state
                if terminated or truncated:
                    self.human_score.finalize_episode()
                    self.human_state, _ = self.human_env.reset()
                    return

    def _step_ai(self) -> None:
        if not self.ai_env or self.ai_state is None or not self.ai_agent:
            return
        action = self.ai_agent.act(self.ai_state)
        next_state, reward, terminated, truncated, _ = self.ai_env.step(action)
        self.ai_score.add_reward(reward)
        if terminated or truncated:
            self.ai_score.finalize_episode()
            self.ai_state, _ = self.ai_env.reset()
        else:
            self.ai_state = next_state
        self._update_ai_value_estimate()

    def _update_ai_value_estimate(self) -> None:
        if (
            self.ai_state is None
            or not self.ai_agent
            or not hasattr(self.ai_agent, "value")
        ):
            self.ai_value_estimate = None
            return
        try:
            self.ai_value_estimate = float(self.ai_agent.value(self.ai_state))
        except Exception:  # noqa: BLE001
            self.ai_value_estimate = None

    def _draw_menu_screen(self) -> None:
        self.screen.fill((12, 12, 30))
        title = self.font_large.render("Lunar Lander", True, (255, 255, 255))
        subtitle = self.font_medium.render("Pick an AI opponent mode", True, (180, 180, 200))

        options = [
            "1. Play vs DQN (neural)",
            "2. Play vs DQN (quantum)",
            "3. Play vs Actor-Critic (quantum)",
            "4. Controls",
            "Escape - Quit the game",
        ]

        width = self.window_width
        height = self.window_height
        center_x = width // 2

        title_y = max(int(height * 0.2), 40)
        self.screen.blit(title, (center_x - title.get_width() // 2, title_y))

        subtitle_y = title_y + title.get_height() + max(20, height // 30)
        self.screen.blit(subtitle, (center_x - subtitle.get_width() // 2, subtitle_y))

        option_start = subtitle_y + subtitle.get_height() + max(30, height // 25)
        option_spacing = self.font_medium.get_height() + max(16, height // 35)

        for idx, line in enumerate(options):
            option_surface = self.font_medium.render(line, True, (200, 200, 200))
            y = option_start + idx * option_spacing
            self.screen.blit(option_surface, (center_x - option_surface.get_width() // 2, y))

        if self.info_message:
            info_surface = self.font_small.render(self.info_message, True, (255, 120, 120))
            info_y = option_start + len(options) * option_spacing + max(20, height // 40)
            info_y = min(height - info_surface.get_height() - 20, info_y)
            self.screen.blit(info_surface, (center_x - info_surface.get_width() // 2, info_y))

    def _draw_controls_screen(self) -> None:
        self.screen.fill((12, 12, 30))
        title = self.font_large.render("Controls", True, (255, 255, 255))
        self.screen.blit(title, ((self.window_width - title.get_width()) // 2, int(self.window_height * 0.15)))

        controls = [
            "Human craft controls:",
            "- Up Arrow: fire main thruster",
            "- Left Arrow: fire right lateral thruster",
            "- Right Arrow: fire left lateral thruster",
            "- Escape: return to menu",
        ]

        y = int(self.window_height * 0.3)
        for line in controls:
            surface = self.font_medium.render(line, True, (200, 200, 200))
            self.screen.blit(surface, ((self.window_width - surface.get_width()) // 2, y))
            y += surface.get_height() + 12

        prompt = self.font_small.render("Press Escape to return to the menu", True, (160, 160, 160))
        self.screen.blit(prompt, ((self.window_width - prompt.get_width()) // 2, self.window_height - 60))


    def _draw_gameplay_screen(self) -> None:
        if not self.ai_env or not self.human_env:
            return

        if self.left_panel_width <= 0 or self.right_panel_width <= 0:
            return

        self.screen.fill((0, 0, 0))

        left_rect = self._fit_panel_rect(0, self.left_panel_width)
        right_rect = self._fit_panel_rect(self.divider_x, self.right_panel_width)

        ai_surface = self._frame_to_surface(
            self.ai_env.render(), left_rect.width, left_rect.height
        )
        human_surface = self._frame_to_surface(
            self.human_env.render(), right_rect.width, right_rect.height
        )

        self.screen.blit(ai_surface, (left_rect.x, left_rect.y))
        self.screen.blit(human_surface, (right_rect.x, right_rect.y))

        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (self.divider_x, 0),
            (self.divider_x, self.panel_height),
            width=4,
        )

        self._draw_score_panel(
            "AI", self.ai_score, 0, self.left_panel_width, self.ai_value_estimate
        )
        self._draw_score_panel(
            "Human", self.human_score, self.divider_x, self.right_panel_width
        )

    def _draw_score_panel(
        self,
        label: str,
        score: EpisodeScore,
        offset_x: int,
        panel_width: int,
        value_estimate: Optional[float] = None,
    ) -> None:
        white = (240, 240, 240)
        accent = (80, 187, 255) if label == "AI" else (255, 196, 87)

        title_surface = self.font_small.render(label, True, white)
        current_surface = self.font_small.render(f"Current: {score.current:7.1f}", True, white)
        last_value = "-" if score.last is None else f"{score.last:7.1f}"
        last_surface = self.font_small.render(f"Last: {last_value}", True, (200, 200, 200))
        best_value = "-" if score.best is None else f"{score.best:7.1f}"
        best_surface = self.font_small.render(f"Best: {best_value}", True, (180, 220, 180))

        surfaces = [title_surface, current_surface, last_surface, best_surface]
        if value_estimate is not None:
            value_surface = self.font_small.render(
                f"Value: {value_estimate:7.2f}", True, (200, 210, 255)
            )
            surfaces.append(value_surface)
        width = max(surface.get_width() for surface in surfaces) + 20
        height = sum(surface.get_height() + 4 for surface in surfaces) + 12

        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        pygame.draw.rect(panel, accent, panel.get_rect(), 2)

        y = 8
        for surface in surfaces:
            panel.blit(surface, (10, y))
            y += surface.get_height() + 4

        target_x = offset_x + panel_width - width - 16
        x = max(offset_x + 10, target_x)
        self.screen.blit(panel, (x, 16))

    def _fit_panel_rect(self, offset_x: int, available_width: int) -> pygame.Rect:
        # Letterbox the 600x400 frame inside the available space without warping it.
        available_width = max(available_width, 1)
        available_height = max(self.panel_height, 1)
        scale = min(available_width / PANEL_WIDTH, available_height / PANEL_HEIGHT)
        scale = max(scale, 0.01)
        width = max(1, int(PANEL_WIDTH * scale))
        height = max(1, int(PANEL_HEIGHT * scale))
        x = offset_x + (available_width - width) // 2
        y = (available_height - height) // 2
        return pygame.Rect(x, y, width, height)

    @staticmethod
    def _frame_to_surface(frame: np.ndarray, width: int, height: int) -> pygame.Surface:
        width = max(width, 1)
        height = max(height, 1)
        swapped = np.transpose(frame, (1, 0, 2))
        surface = pygame.surfarray.make_surface(swapped)
        return pygame.transform.smoothscale(surface, (width, height))

    @staticmethod
    @contextmanager
    def _scaled_human_thrusters():
        # Temporarily attenuate the human thrusters so they feel less twitchy.
        original_main = lunar_lander.MAIN_ENGINE_POWER
        original_side = lunar_lander.SIDE_ENGINE_POWER
        lunar_lander.MAIN_ENGINE_POWER = original_main * HUMAN_MAIN_THRUST_SCALE
        lunar_lander.SIDE_ENGINE_POWER = original_side * HUMAN_SIDE_THRUST_SCALE
        try:
            yield
        finally:
            lunar_lander.MAIN_ENGINE_POWER = original_main
            lunar_lander.SIDE_ENGINE_POWER = original_side


def main() -> None:
    GameApp().run()


if __name__ == "__main__":
    main()
