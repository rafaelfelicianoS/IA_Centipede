"""
Autonomous Agent for Centipede Game
Authors: AI Centipede Team
Strategy: Multi-layered intelligent agent with prediction, pathfinding, and adaptive tactics
"""

import asyncio
import getpass
import json
import os
import math
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
import logging

import pygame
import websockets

# Import our advanced analysis modules
try:
    from agent_analysis import (
        CentipedePredictor,
        MushroomAnalyzer,
        ThreatAnalyzer,
        ThreatAssessment
    )
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    logging.warning("Analysis modules not available, using basic strategies")

# Configure logging: place log next to this file and use UTF-8
LOG_DIR = os.path.dirname(__file__)
LOG_PATH = os.path.join(LOG_DIR, 'agent_debug.log')
file_handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger('CentipedeAgent')

try:
    pygame.init()
    program_icon = pygame.image.load("data/icon2.png")
    pygame.display.set_icon(program_icon)
except Exception:
    logger.debug("pygame init or icon load failed; continuing without icon")


@dataclass
class Position:
    """Represents a position on the game map"""
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Manhattan distance to another position"""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def euclidean_distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class Centipede:
    """Represents a centipede in the game"""
    name: str
    body: List[Tuple[int, int]]
    direction: int  # 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST
    
    @property
    def head(self) -> Position:
        """Get the head position"""
        if self.body:
            return Position(self.body[0][0], self.body[0][1])
        return Position(0, 0)
    
    @property
    def length(self) -> int:
        return len(self.body)
    
    def predict_next_positions(self, num_steps: int = 5) -> List[Position]:
        """Predict where the centipede head will be in the next steps"""
        # Simplified prediction - assumes horizontal movement
        predictions = []
        current = self.head
        
        # Determine horizontal direction
        if self.direction == 1:  # EAST
            for i in range(1, num_steps + 1):
                predictions.append(Position(current.x + i, current.y))
        elif self.direction == 3:  # WEST
            for i in range(1, num_steps + 1):
                predictions.append(Position(current.x - i, current.y))
        
        return predictions


class GameState:
    """Manages the current state of the game"""
    
    def __init__(self):
        self.map_width = 40
        self.map_height = 24
        self.bug_blaster_pos: Optional[Position] = None
        self.centipedes: List[Centipede] = []
        self.mushrooms: Set[Position] = set()
        self.blasts: Set[Position] = set()
        self.score = 0
        self.step = 0
        self.safe_zone_y = 19  # Bottom 5 rows (24-5=19)
        
        # Threat tracking
        self.danger_zones: Set[Position] = set()
        self.last_action = ""
        self.cooldown = 0
        
        # History tracking for pattern detection
        self.centipede_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.position_history: deque = deque(maxlen=50)
        self.score_history: deque = deque(maxlen=100)
        
        # Analysis systems (if available)
        if ANALYSIS_AVAILABLE:
            self.predictor = CentipedePredictor(self.map_width, self.map_height)
            self.mushroom_analyzer = MushroomAnalyzer(self.map_width, self.map_height)
            self.threat_analyzer = ThreatAnalyzer(self.map_width, self.map_height)
        else:
            self.predictor = None
            self.mushroom_analyzer = None
            self.threat_analyzer = None
        
        # Performance metrics
        self.hits_made = 0
        self.shots_fired = 0
        self.mushrooms_destroyed = 0
        
        # Futile chase detection
        self.shot_attempts: Dict[Tuple[int, int], int] = {}  # (x, y) -> count
        self.last_shot_target: Optional[Tuple[int, int]] = None
        self.consecutive_misses = 0
        self.chase_cooldowns: Dict[str, int] = {}  # centipede_name -> step to resume
        
        # Player stuck detection
        self.last_positions: deque = deque(maxlen=10)  # Track last 10 positions
        self.stuck_counter = 0  # How many frames player hasn't moved
        self.last_action_attempted = ""  # Last movement command tried
        
    def update(self, state: dict):
        """Update game state from server message"""
        prev_pos = self.bug_blaster_pos
        
        if 'bug_blaster' in state:
            pos = state['bug_blaster']['pos']
            new_pos = Position(pos[0], pos[1])
            self.bug_blaster_pos = new_pos
            self.position_history.append(new_pos.to_tuple())
            
            # Track stuck detection
            self.last_positions.append(new_pos.to_tuple())
            
            # Check if player is stuck (position hasn't changed)
            if prev_pos and prev_pos == new_pos:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        if 'centipedes' in state:
            self.centipedes = [
                Centipede(
                    name=c['name'],
                    body=c['body'],
                    direction=c['direction']
                )
                for c in state['centipedes']
            ]
            
            # Track centipede movements
            for centipede in self.centipedes:
                self.centipede_history[centipede.name].append(centipede.head.to_tuple())
        
        if 'mushrooms' in state:
            old_mushroom_count = len(self.mushrooms)
            self.mushrooms = {
                Position(m['pos'][0], m['pos'][1])
                for m in state['mushrooms']
            }
            # Track mushrooms destroyed
            if old_mushroom_count > len(self.mushrooms):
                self.mushrooms_destroyed += old_mushroom_count - len(self.mushrooms)
        
        if 'blasts' in state:
            self.blasts = {
                Position(b[0], b[1])
                for b in state['blasts']
            }
        
        prev_score = self.score
        self.score = state.get('score', 0)
        self.step = state.get('step', 0)
        self.score_history.append(self.score)
        
        # Track hits
        if self.score > prev_score:
            score_gain = self.score - prev_score
            if score_gain >= 100:  # Hit a centipede
                self.hits_made += 1
                logger.info(f"HIT! Score +{score_gain}. Total hits: {self.hits_made}")
                # Reset futile chase tracking on successful hit
                self.shot_attempts.clear()
                self.consecutive_misses = 0
                self.last_shot_target = None
        
        # Update danger zones
        self._update_danger_zones()
    
    def _update_danger_zones(self):
        """Calculate positions that are dangerous"""
        self.danger_zones.clear()
        
        # Mark all centipede body positions as dangerous
        for centipede in self.centipedes:
            for seg in centipede.body:
                self.danger_zones.add(Position(seg[0], seg[1]))
                # Mark positions around centipede segments
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = seg[0] + dx, seg[1] + dy
                        if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                            self.danger_zones.add(Position(nx, ny))
    
    def is_safe_position(self, pos: Position, extra_danger: Optional[Set[Position]] = None) -> bool:
        """Check if a position is safe to move to"""
        # Check boundaries
        if not (0 <= pos.x < self.map_width and 0 <= pos.y < self.map_height):
            return False
        
        # Check if position has mushroom
        if pos in self.mushrooms:
            return False
        
        # Check if position is in danger zone
        if pos in self.danger_zones:
            return False
        
        # Check extra dangers if provided
        if extra_danger and pos in extra_danger:
            return False
        
        return True
    
    def get_nearest_centipede_head(self) -> Optional[Centipede]:
        """Get the centipede head nearest to bug blaster"""
        if not self.centipedes or not self.bug_blaster_pos:
            return None
        
        nearest = min(
            self.centipedes,
            key=lambda c: self.bug_blaster_pos.distance_to(c.head)
        )
        return nearest
    
    def count_mushrooms_in_column(self, x: int) -> int:
        """Count mushrooms in a specific column"""
        return sum(1 for m in self.mushrooms if m.x == x)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        accuracy = (self.hits_made / self.shots_fired * 100) if self.shots_fired > 0 else 0
        return {
            'hits_made': self.hits_made,
            'shots_fired': self.shots_fired,
            'accuracy': accuracy,
            'mushrooms_destroyed': self.mushrooms_destroyed,
            'score_per_step': self.score / max(self.step, 1)
        }
    
    def detect_stuck_centipedes(self) -> List[str]:
        """Detect centipedes that appear to be stuck"""
        stuck_centipedes = []
        
        if self.predictor:
            for centipede in self.centipedes:
                history = list(self.centipede_history[centipede.name])
                if self.predictor.detect_stuck_centipede(
                    centipede.body,
                    history,
                    threshold=10
                ):
                    stuck_centipedes.append(centipede.name)
                    logger.warning(f"Centipede {centipede.name} appears stuck!")
        
        return stuck_centipedes
    
    def is_player_stuck(self, threshold: int = 5) -> bool:
        """
        Detect if player is stuck (trying to move but position not changing)
        Returns True if player hasn't moved for 'threshold' consecutive frames
        """
        if self.stuck_counter >= threshold:
            return True
        
        # Alternative: check if last N positions are all the same
        if len(self.last_positions) >= threshold:
            unique_positions = set(list(self.last_positions)[-threshold:])
            if len(unique_positions) <= 1:  # All same position
                return True
        
        return False
        


class PathFinder:
    """A* pathfinding for safe navigation"""
    
    def __init__(self, game_state: GameState):
        self.game_state = game_state
    
    def find_path(self, start: Position, goal: Position) -> Optional[List[Position]]:
        """Find safe path from start to goal using A*"""
        if not self.game_state.is_safe_position(goal):
            return None
        
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            frontier.sort(key=lambda x: x[0])
            _, current = frontier.pop(0)
            
            if current == goal:
                break
            
            for next_pos in self._get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + goal.distance_to(next_pos)
                    frontier.append((priority, next_pos))
                    came_from[next_pos] = current
        
        if goal not in came_from:
            return None
        
        # Reconstruct path
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path
    
    def _get_neighbors(self, pos: Position) -> List[Position]:
        """Get valid neighboring positions"""
        neighbors = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Up, Down, Left, Right
            next_pos = Position(pos.x + dx, pos.y + dy)
            if self.game_state.is_safe_position(next_pos):
                neighbors.append(next_pos)
        return neighbors


class TargetingSystem:
    """Intelligent targeting system for shooting centipedes"""
    
    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.last_shot_step = 0
    
    def _is_futile_chase(self, target: Tuple[int, int], centipede: Centipede) -> bool:
        """
        Detect if shooting at this target is a futile chase.
        A chase is futile when:
        - We've shot at similar positions multiple times without hitting
        - The centipede is moving horizontally away from us
        - Player and centipede have same speed, so we can never catch up
        """
        # Check if this centipede is on chase cooldown
        if centipede.name in self.game_state.chase_cooldowns:
            cooldown_until = self.game_state.chase_cooldowns[centipede.name]
            if self.game_state.step < cooldown_until:
                logger.debug(f"Centipede {centipede.name} on chase cooldown until step {cooldown_until}")
                return True
        
        # Track shot attempts at this position
        if self.game_state.last_shot_target:
            # Check if we're chasing horizontally
            last_x, last_y = self.game_state.last_shot_target
            curr_x, curr_y = target
            
            # Horizontal chase: x is changing, y is similar
            if abs(curr_y - last_y) <= 2:  # Same row or nearby
                # If we've been chasing for several shots without hitting
                if self.game_state.consecutive_misses >= 3:
                    # Check if centipede is moving away horizontally
                    if len(centipede.body) >= 2:
                        head_x, head_y = centipede.body[0]
                        blaster_x = self.game_state.bug_blaster_pos.x
                        
                        # Is centipede moving away from us horizontally?
                        distance_increasing = abs(head_x - blaster_x) > 3
                        
                        if distance_increasing:
                            # This is a futile chase!
                            logger.warning(f"FUTILE CHASE detected! Centipede {centipede.name} at {target}, "
                                         f"consecutive misses: {self.game_state.consecutive_misses}")
                            # Put centipede on cooldown
                            self.game_state.chase_cooldowns[centipede.name] = self.game_state.step + 15
                            return True
        
        return False
    
    def _record_shot_attempt(self, target: Tuple[int, int]):
        """Record a shot attempt at a target"""
        # Update consecutive misses tracking
        if self.game_state.last_shot_target:
            last_x, last_y = self.game_state.last_shot_target
            curr_x, curr_y = target
            
            # If shooting at similar position (horizontal chase)
            if abs(curr_y - last_y) <= 2:
                self.game_state.consecutive_misses += 1
            else:
                # Different row, reset counter
                self.game_state.consecutive_misses = 0
        
        self.game_state.last_shot_target = target
        
        # Track attempts at this position
        if target in self.game_state.shot_attempts:
            self.game_state.shot_attempts[target] += 1
        else:
            self.game_state.shot_attempts[target] = 1
    
    def should_shoot(self) -> Tuple[bool, str]:
        """
        Determine if we should shoot now
        Returns (should_shoot, reason)
        """
        if not self.game_state.bug_blaster_pos:
            return (False, "No blaster position")
        
        blaster_x = self.game_state.bug_blaster_pos.x
        blaster_y = self.game_state.bug_blaster_pos.y
        
        # Check cooldown - reduced to 1 for more aggressive shooting
        if self.game_state.step - self.last_shot_step < 1:
            return (False, "Cooldown")
        
        best_shot_value = 0
        best_shot_target = None
        best_shot_centipede = None
        
        # Check each centipede segment
        for centipede in self.game_state.centipedes:
            # Skip if this centipede is on chase cooldown
            if centipede.name in self.game_state.chase_cooldowns:
                if self.game_state.step < self.game_state.chase_cooldowns[centipede.name]:
                    continue
            
            for i, (seg_x, seg_y) in enumerate(centipede.body):
                if seg_x == blaster_x and seg_y < blaster_y:
                    # Check if path is clear (no mushrooms in the way)
                    clear_path = True
                    for y in range(seg_y + 1, blaster_y):
                        if Position(blaster_x, y) in self.game_state.mushrooms:
                            clear_path = False
                            break
                    
                    if clear_path:
                        # Calculate shot value
                        # Higher value for: closer, head shots, higher on screen
                        distance = blaster_y - seg_y
                        is_head = (i == 0 or i == len(centipede.body) - 1)
                        height_bonus = (self.game_state.map_height - seg_y) * 2
                        
                        shot_value = 100 - distance + height_bonus
                        if is_head:
                            shot_value += 50
                        
                        if shot_value > best_shot_value:
                            best_shot_value = shot_value
                            best_shot_target = (seg_x, seg_y)
                            best_shot_centipede = centipede
        
        if best_shot_value > 20 and best_shot_target and best_shot_centipede:
            # Check if this would be a futile chase
            if self._is_futile_chase(best_shot_target, best_shot_centipede):
                return (False, f"Futile chase avoided at {best_shot_target}")
            
            # Record the shot attempt
            self._record_shot_attempt(best_shot_target)
            self.last_shot_step = self.game_state.step
            self.game_state.shots_fired += 1
            return (True, f"Good shot at {best_shot_target}, value={best_shot_value}")
        
        # Use predictor if available
        if self.game_state.predictor and self.game_state.centipedes:
            for centipede in self.game_state.centipedes:
                mushroom_positions = {(m.x, m.y) for m in self.game_state.mushrooms}
                predictions = self.game_state.predictor.predict_trajectory(
                    centipede.body,
                    centipede.direction,
                    mushroom_positions,
                    num_steps=5
                )
                
                # Check if centipede will pass through our column
                for step_num, future_body in enumerate(predictions):
                    for seg_x, seg_y in future_body:
                        if seg_x == blaster_x and seg_y < blaster_y:
                            # Predictive shot
                            if step_num <= 2:  # Close prediction
                                self.last_shot_step = self.game_state.step
                                self.game_state.shots_fired += 1
                                return (True, f"Predictive shot, step={step_num}")
        
        return (False, "No good shot")
    
    def get_best_shooting_position(self) -> Optional[Position]:
        """Find the best position to shoot from"""
        if not self.game_state.centipedes:
            return None
        
        # Use advanced analysis if available
        if self.game_state.predictor:
            mushroom_positions = {(m.x, m.y) for m in self.game_state.mushrooms}
            all_predictions = []
            
            for centipede in self.game_state.centipedes:
                predictions = self.game_state.predictor.predict_trajectory(
                    centipede.body,
                    centipede.direction,
                    mushroom_positions,
                    num_steps=10
                )
                all_predictions.append(predictions)
            
            # Find shooting lanes
            lanes = self.game_state.predictor.find_safe_shooting_lanes(
                (self.game_state.bug_blaster_pos.x, self.game_state.bug_blaster_pos.y),
                all_predictions,
                mushroom_positions
            )
            
            if lanes:
                best_lane = lanes[0]
                logger.debug(f"Best shooting lane: x={best_lane[0]}, probability={best_lane[2]:.2f}")
                return Position(best_lane[0], self.game_state.bug_blaster_pos.y)
        
        # Fallback: aim at nearest centipede
        nearest = self.game_state.get_nearest_centipede_head()
        if nearest:
            return Position(nearest.head.x, self.game_state.bug_blaster_pos.y)
        
        return None
    
    def find_mushroom_to_clear(self) -> Optional[Position]:
        """Find best mushroom to clear for strategic advantage"""
        if not self.game_state.mushroom_analyzer or not self.game_state.bug_blaster_pos:
            # Fallback: find nearest mushroom in safe zone
            safe_mushrooms = [
                m for m in self.game_state.mushrooms
                if m.y >= self.game_state.safe_zone_y
            ]
            if safe_mushrooms:
                return min(
                    safe_mushrooms,
                    key=lambda m: self.game_state.bug_blaster_pos.distance_to(m)
                )
            return None
        
        # Use advanced analysis
        blaster_tuple = (self.game_state.bug_blaster_pos.x, self.game_state.bug_blaster_pos.y)
        centipede_positions = [
            pos for c in self.game_state.centipedes for pos in c.body
        ]
        
        # Calculate priority for each mushroom
        mushroom_priorities = []
        for mushroom in self.game_state.mushrooms:
            if mushroom.y >= self.game_state.safe_zone_y - 5:  # In or near safe zone
                priority = self.game_state.mushroom_analyzer.calculate_clear_priority(
                    (mushroom.x, mushroom.y),
                    blaster_tuple,
                    centipede_positions
                )
                mushroom_priorities.append((mushroom, priority))
        
        if mushroom_priorities:
            mushroom_priorities.sort(key=lambda mp: mp[1], reverse=True)
            return mushroom_priorities[0][0]
        
        return None


class StrategyManager:
    """Manages overall strategy and decision making"""
    
    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.pathfinder = PathFinder(game_state)
        self.targeting = TargetingSystem(game_state)
        self.last_safe_pos: Optional[Position] = None
        
        # Strategy state
        self.current_strategy = "AGGRESSIVE"  # AGGRESSIVE, DEFENSIVE, CLEARING, POSITIONING
        self.strategy_timer = 0
        self.stuck_counter = 0
        
    def decide_action(self) -> str:
        """Main decision function - returns action command"""
        if not self.game_state.bug_blaster_pos:
            return ""
        
        current_pos = self.game_state.bug_blaster_pos
        
        # Log performance stats periodically
        if self.game_state.step % 100 == 0:
            stats = self.game_state.get_performance_stats()
            logger.info(f"Performance: {stats}")
            
            # Detect stuck centipedes
            stuck = self.game_state.detect_stuck_centipedes()
            if stuck:
                logger.warning(f"Stuck centipedes detected: {stuck}")
        
        # Update strategy based on game state
        self._update_strategy()
        
        # PRIORITY -1: Unstuck maneuver
        # If player is stuck AND there are stuck centipedes, try alternative movement
        if self.game_state.is_player_stuck(threshold=5):
            stuck_centipedes = self.game_state.detect_stuck_centipedes()
            if stuck_centipedes:
                logger.warning(f"PLAYER AND CENTIPEDES STUCK! Player stuck for {self.game_state.stuck_counter} frames")
                unstuck_action = self._unstuck_maneuver(current_pos)
                if unstuck_action:
                    logger.info(f"Unstuck maneuver: {unstuck_action}")
                    return unstuck_action
        
        # PRIORITY 0: Immediate shot if perfectly aligned (before emergency check)
        # This ensures we never miss a perfect shot even when in slight danger
        # BUT we need to avoid futile horizontal chases
        if self.game_state.centipedes:
            for centipede in self.game_state.centipedes:
                # Skip centipedes on chase cooldown
                if centipede.name in self.game_state.chase_cooldowns:
                    if self.game_state.step < self.game_state.chase_cooldowns[centipede.name]:
                        continue
                
                for seg_x, seg_y in centipede.body:
                    if seg_x == current_pos.x and seg_y < current_pos.y:
                        # Check if path is perfectly clear
                        clear = True
                        for y in range(seg_y + 1, current_pos.y):
                            if Position(current_pos.x, y) in self.game_state.mushrooms:
                                clear = False
                                break
                        if clear and (current_pos.y - seg_y) <= 5:  # Close enough
                            target = (seg_x, seg_y)
                            # Check if this would be a futile chase
                            if self.targeting._is_futile_chase(target, centipede):
                                logger.debug(f"Skipping PERFECT SHOT at {seg_x},{seg_y} - futile chase")
                                continue
                            
                            # Record shot attempt and shoot
                            self.targeting._record_shot_attempt(target)
                            logger.debug(f"PERFECT SHOT at {seg_x},{seg_y}")
                            return "A"
        
        # PRIORITY 0.5: Safe vertical shot - shoot centipede directly above even if in danger
        # This is based on game mechanics: when we shoot a centipede above us,
        # it spawns a mushroom and turns away, so we're safe if we don't move
        can_shoot_safe, target = self._can_safely_shoot_vertical(current_pos)
        if can_shoot_safe:
            logger.info(f"SAFE VERTICAL SHOT! Shooting centipede at {target} instead of evading")
            return "A"
        
        # PRIORITY 1: Emergency evasion
        # Only evade if we DON'T have a safe vertical shot available
        if self._is_in_immediate_danger(current_pos):
            logger.warning(f"EMERGENCY! Position {current_pos.to_tuple()} is dangerous!")
            # Double-check: do we have a safe shot before evading?
            can_shoot_safe, target = self._can_safely_shoot_vertical(current_pos)
            if can_shoot_safe:
                logger.info(f"EMERGENCY OVERRIDE: Safe shot available at {target}, shooting instead of evading!")
                return "A"
            
            escape_action = self._emergency_escape()
            if escape_action:
                logger.info(f"Emergency escape: {escape_action}")
                return escape_action
        
        # PRIORITY 2: Shoot if we have a clear shot (but not in danger)
        should_shoot, reason = self.targeting.should_shoot()
        if should_shoot:
            logger.debug(f"Shooting: {reason}")
            return "A"
        
        # PRIORITY 2.5: Return to safe zone if we're too far up
        # This is CRITICAL: we can only score points when below centipedes!
        # If we're above safe_zone_y and there's no immediate danger, move down
        if current_pos.y < self.game_state.safe_zone_y:
            # Check if there are centipedes nearby that we need to avoid
            min_centipede_dist = float('inf')
            if self.game_state.centipedes:
                for centipede in self.game_state.centipedes:
                    dist = current_pos.distance_to(centipede.head)
                    min_centipede_dist = min(min_centipede_dist, dist)
            
            # Only return to safe zone if centipedes are far enough (not immediate threat)
            if min_centipede_dist > 5:  # Safe distance
                # Check if moving down is safe
                down_pos = Position(current_pos.x, current_pos.y + 1)
                if self.game_state.is_safe_position(down_pos):
                    logger.info(f"RETURNING TO SAFE ZONE: Currently at y={current_pos.y}, moving down to y={down_pos.y}")
                    return 's'
                # If can't move straight down, try diagonal down
                else:
                    for dx in [-1, 1]:  # Try left-down or right-down
                        diag_pos = Position(current_pos.x + dx, current_pos.y + 1)
                        if self.game_state.is_safe_position(diag_pos):
                            logger.debug(f"Returning to safe zone diagonally: {'a' if dx < 0 else 'd'} then s")
                            return 'a' if dx < 0 else 'd'
        
        # Execute current strategy
        if self.current_strategy == "AGGRESSIVE":
            action = self._aggressive_strategy()
        elif self.current_strategy == "DEFENSIVE":
            action = self._defensive_strategy()
        elif self.current_strategy == "CLEARING":
            action = self._clearing_strategy()
        elif self.current_strategy == "POSITIONING":
            action = self._positioning_strategy()
        else:
            action = self._default_strategy()
        
        if action:
            return action
        
        # FALLBACK: Smart patrol
        return self._smart_patrol()
    
    def _update_strategy(self):
        """Update current strategy based on game state"""
        self.strategy_timer += 1
        
        # Count nearby threats
        threat_count = sum(
            1 for d in self.game_state.danger_zones
            if d.distance_to(self.game_state.bug_blaster_pos) <= 5
        )
        
        # Count centipedes
        centipede_count = len(self.game_state.centipedes)
        
        # Check if centipedes are nearby - AUMENTADO O RANGE para 15
        centipedes_nearby = False
        if self.game_state.centipedes:
            nearest = self.game_state.get_nearest_centipede_head()
            if nearest:
                dist = self.game_state.bug_blaster_pos.distance_to(nearest.head)
                if dist < 15:  # AUMENTADO de 10 para 15
                    centipedes_nearby = True
                    logger.debug(f"Centipede nearby at distance {dist}")
        
        # Count mushrooms in movement area
        mushroom_count = sum(
            1 for m in self.game_state.mushrooms
            if m.y >= self.game_state.safe_zone_y
        )
        
        # Decide strategy - PRIORIDADE MUDADA!
        # ALTA PRIORIDADE: Se há centopeia, seja agressivo!
        if centipedes_nearby and centipede_count > 0:
            # If centipedes are nearby, be aggressive! (MOVED TO TOP PRIORITY)
            if self.current_strategy != "AGGRESSIVE":
                logger.info("Switching to AGGRESSIVE strategy - centipedes nearby!")
            self.current_strategy = "AGGRESSIVE"
            self.strategy_timer = 0
        
        elif threat_count > 10:
            if self.current_strategy != "DEFENSIVE":
                logger.info("Switching to DEFENSIVE strategy")
            self.current_strategy = "DEFENSIVE"
            self.strategy_timer = 0
        
        elif mushroom_count > 20 and threat_count < 5 and not centipedes_nearby:
            if self.current_strategy != "CLEARING":
                logger.info("Switching to CLEARING strategy")
            self.current_strategy = "CLEARING"
            self.strategy_timer = 0
        
        elif self.game_state.step < 300 and not centipedes_nearby and centipede_count == 0:
            # Early game: position well ONLY if NO centipedes exist at all
            if self.current_strategy != "POSITIONING":
                logger.info("Switching to POSITIONING strategy")
            self.current_strategy = "POSITIONING"
            self.strategy_timer = 0
        
        else:
            # Default to aggressive when in doubt
            if self.current_strategy != "AGGRESSIVE":
                logger.info("Switching to AGGRESSIVE strategy (default)")
            self.current_strategy = "AGGRESSIVE"
            self.strategy_timer = 0
    
    def _aggressive_strategy(self) -> Optional[str]:
        """Aggressive strategy: actively hunt centipedes"""
        current_pos = self.game_state.bug_blaster_pos
        
        # HIGHEST PRIORITY: Shoot at ANY centipede segment if aligned
        # BUT avoid futile horizontal chases
        if self.game_state.centipedes:
            for centipede in self.game_state.centipedes:
                # Skip centipedes on chase cooldown
                if centipede.name in self.game_state.chase_cooldowns:
                    if self.game_state.step < self.game_state.chase_cooldowns[centipede.name]:
                        continue
                
                for i, (seg_x, seg_y) in enumerate(centipede.body):
                    if seg_x == current_pos.x and seg_y < current_pos.y:
                        # Check if path is clear
                        clear = True
                        for y in range(seg_y + 1, current_pos.y):
                            if Position(current_pos.x, y) in self.game_state.mushrooms:
                                clear = False
                                break
                        if clear:
                            target = (seg_x, seg_y)
                            # Check if this would be a futile chase
                            if self.targeting._is_futile_chase(target, centipede):
                                logger.debug(f"Skipping aggressive shot at {seg_x},{seg_y} - futile chase")
                                continue
                            
                            is_head = (i == 0 or i == len(centipede.body) - 1)
                            self.targeting._record_shot_attempt(target)
                            logger.debug(f"Aggressive shot at {'head' if is_head else 'segment'} at {seg_x},{seg_y}")
                            return "A"
        
        # Check if we should shoot at nearest centipede (even if not perfectly aligned)
        if self.game_state.centipedes:
            nearest_centipede = self.game_state.get_nearest_centipede_head()
            if nearest_centipede:
                head = nearest_centipede.head
                # If centipede is above us and close to our column, try to shoot
                if abs(head.x - current_pos.x) <= 1 and head.y < current_pos.y:
                    # Check if path is mostly clear
                    mushrooms_in_path = sum(
                        1 for m in self.game_state.mushrooms
                        if m.x == current_pos.x and m.y < current_pos.y and m.y > head.y
                    )
                    if mushrooms_in_path == 0:  # Strict: path must be clear
                        logger.debug(f"Aggressive shot at centipede head at {head.to_tuple()}")
                        return "A"
        
        # Move to best shooting position
        best_pos = self.targeting.get_best_shooting_position()
        if best_pos and best_pos != current_pos:
            # But check if it's safe
            if not self._is_in_immediate_danger(best_pos):
                move = self._get_move_towards(current_pos, best_pos)
                if move:
                    logger.debug(f"Aggressive move towards {best_pos.to_tuple()}")
                    return move
        
        # If centipede is nearby but not in our column, move towards it
        if self.game_state.centipedes:
            nearest = self.game_state.get_nearest_centipede_head()
            if nearest and nearest.head.y < current_pos.y:
                # Try to align with centipede
                if abs(nearest.head.x - current_pos.x) > 1:
                    target = Position(nearest.head.x, current_pos.y)
                    if self.game_state.is_safe_position(target):
                        move = self._get_move_towards(current_pos, target)
                        if move:
                            logger.debug(f"Moving to align with centipede at x={nearest.head.x}")
                            return move
        
        # If can't move to shooting position, stay mobile
        return self._stay_mobile()
    
    def _defensive_strategy(self) -> Optional[str]:
        """Defensive strategy: prioritize survival but shoot when safe"""
        current_pos = self.game_state.bug_blaster_pos
        
        # Even in defensive mode, shoot if we have a clear, safe shot
        if self.game_state.centipedes:
            for centipede in self.game_state.centipedes:
                for seg_x, seg_y in centipede.body:
                    # Only shoot if centipede is far enough (safe)
                    if seg_x == current_pos.x and seg_y < current_pos.y and (current_pos.y - seg_y) > 3:
                        # Check for clear shot
                        clear = True
                        for y in range(seg_y + 1, current_pos.y):
                            if Position(current_pos.x, y) in self.game_state.mushrooms:
                                clear = False
                                break
                        if clear:
                            logger.debug(f"Defensive: safe shot at {seg_x},{seg_y}")
                            return "A"
        
        # Find safest position in bottom rows
        safest_pos = self._find_safest_position_in_safe_zone()
        if safest_pos:
            move = self._get_move_towards(current_pos, safest_pos)
            if move:
                logger.debug(f"Defensive move towards {safest_pos.to_tuple()}")
                return move
        
        return self._stay_mobile()
    
    def _clearing_strategy(self) -> Optional[str]:
        """Clear mushrooms to create better tactical positions"""
        current_pos = self.game_state.bug_blaster_pos
        
        # Find mushroom to clear
        mushroom = self.targeting.find_mushroom_to_clear()
        if mushroom:
            # Position to shoot mushroom
            if mushroom.x == current_pos.x and mushroom.y < current_pos.y:
                # Can shoot it
                logger.debug(f"Clearing mushroom at {mushroom.to_tuple()}")
                return "A"
            else:
                # Move to align
                target = Position(mushroom.x, current_pos.y)
                if self.game_state.is_safe_position(target):
                    move = self._get_move_towards(current_pos, target)
                    if move:
                        return move
        
        return None
    
    def _positioning_strategy(self) -> Optional[str]:
        """Position for optimal play"""
        current_pos = self.game_state.bug_blaster_pos
        
        # If centipedes appear, shoot them first!
        if self.game_state.centipedes:
            # Check ALL centipede segments, not just heads
            for centipede in self.game_state.centipedes:
                for seg_x, seg_y in centipede.body:
                    if seg_x == current_pos.x and seg_y < current_pos.y:
                        # Check for clear shot
                        clear = True
                        for m in self.game_state.mushrooms:
                            if m.x == current_pos.x and seg_y < m.y < current_pos.y:
                                clear = False
                                break
                        if clear:
                            logger.debug(f"Positioning strategy: opportunistic shot at {seg_x},{seg_y}")
                            return "A"
        
        # Move to center of safe zone for flexibility
        ideal_x = self.game_state.map_width // 2
        ideal_y = self.game_state.map_height - 3
        
        ideal_pos = Position(ideal_x, ideal_y)
        
        # CORREÇÃO: Verificar se já está perto o suficiente
        distance = current_pos.distance_to(ideal_pos)
        if distance > 2:
            move = self._get_move_towards(current_pos, ideal_pos)
            if move:
                logger.debug(f"Positioning for optimal play (distance={distance})")
                return move
            else:
                # Se não consegue se mover em direção à posição ideal,
                # mude para patrulha inteligente
                logger.debug("Can't reach ideal position, switching to smart patrol")
                return self._smart_patrol()
        
        # Se já está na posição ideal, patrulhe
        logger.debug("Already at ideal position, patrolling")
        return self._smart_patrol()
    
    def _default_strategy(self) -> Optional[str]:
        """Default balanced strategy"""
        return self._aggressive_strategy()
    
    def _can_safely_shoot_vertical(self, pos: Position) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        Check if we can safely shoot at a centipede directly above us.
        IMPORTANT: This is only safe if there are NO centipedes approaching horizontally!
        
        Returns: (can_shoot, target_position)
        """
        if not self.game_state.centipedes:
            return (False, None)
        
        # CRITICAL FIX: Check for horizontal threats FIRST
        # If we're going to shoot and stay still, we need to make sure no centipede 
        # can hit us horizontally while we're shooting
        for centipede in self.game_state.centipedes:
            for seg_x, seg_y in centipede.body:
                # Check if centipede is on our row or 1 row above/below
                if abs(seg_y - pos.y) <= 1:
                    # Check if it's close horizontally (within 3 tiles)
                    if abs(seg_x - pos.x) <= 3:
                        # Too dangerous - centipede could hit us while we're shooting!
                        logger.debug(f"UNSAFE vertical shot: centipede at ({seg_x},{seg_y}) too close horizontally")
                        return (False, None)
        
        # Now check for vertical shot opportunities
        for centipede in self.game_state.centipedes:
            for seg_x, seg_y in centipede.body:
                # Is it directly above us?
                if seg_x == pos.x and seg_y < pos.y:
                    # Check if path is clear (no mushrooms blocking)
                    clear = True
                    for y in range(seg_y + 1, pos.y):
                        if Position(pos.x, y) in self.game_state.mushrooms:
                            clear = False
                            break
                    
                    if clear:
                        # This is a safe shot!
                        # When we hit: mushroom spawns, centipede turns/splits
                        # If we don't move after shooting, we're safe (assuming no horizontal threats)
                        logger.info(f"SAFE VERTICAL SHOT available at ({seg_x},{seg_y}), distance={pos.y - seg_y}")
                        return (True, (seg_x, seg_y))
        
        return (False, None)
        
        # Check each centipede for segments directly above us
        for centipede in self.game_state.centipedes:
            for seg_x, seg_y in centipede.body:
                # Is it directly above us?
                if seg_x == pos.x and seg_y < pos.y:
                    # Check if path is clear (no mushrooms blocking)
                    clear = True
                    for y in range(seg_y + 1, pos.y):
                        if Position(pos.x, y) in self.game_state.mushrooms:
                            clear = False
                            break
                    
                    if clear:
                        # This is a safe shot!
                        # When we hit: mushroom spawns, centipede turns/splits
                        # If we don't move after shooting, we're safe
                        logger.info(f"SAFE VERTICAL SHOT available at ({seg_x},{seg_y}), distance={pos.y - seg_y}")
                        return (True, (seg_x, seg_y))
        
        return (False, None)
    
    def _is_in_immediate_danger(self, pos: Position) -> bool:
        """Check if position is in immediate danger"""
        # Check if in danger zone
        if pos in self.game_state.danger_zones:
            return True
        
        # Check proximity to centipede segments
        for centipede in self.game_state.centipedes:
            for seg_x, seg_y in centipede.body:
                if abs(pos.x - seg_x) <= 1 and abs(pos.y - seg_y) <= 1:
                    return True
        
        # Use threat analyzer if available
        if self.game_state.threat_analyzer:
            centipede_bodies = [c.body for c in self.game_state.centipedes]
            mushroom_tuples = {(m.x, m.y) for m in self.game_state.mushrooms}
            
            # Get predictions
            all_predictions = []
            if self.game_state.predictor:
                for centipede in self.game_state.centipedes:
                    predictions = self.game_state.predictor.predict_trajectory(
                        centipede.body,
                        centipede.direction,
                        mushroom_tuples,
                        num_steps=5
                    )
                    all_predictions.append(predictions)
            
            assessment = self.game_state.threat_analyzer.assess_position(
                (pos.x, pos.y),
                centipede_bodies,
                all_predictions,
                mushroom_tuples
            )
            
            if assessment.threat_level >= 0.7:
                logger.warning(f"High threat at {pos.to_tuple()}: {assessment.threat_sources}")
                return True
        
        return False
    
    def _emergency_escape(self) -> Optional[str]:
        """Find quickest escape from danger"""
        current = self.game_state.bug_blaster_pos
        
        # Use threat analyzer if available
        if self.game_state.threat_analyzer:
            danger_tuples = {(d.x, d.y) for d in self.game_state.danger_zones}
            mushroom_tuples = {(m.x, m.y) for m in self.game_state.mushrooms}
            
            safe_positions = self.game_state.threat_analyzer.find_safe_retreat_positions(
                (current.x, current.y),
                danger_tuples,
                mushroom_tuples,
                max_distance=3
            )
            
            if safe_positions:
                best_retreat = safe_positions[0][0]
                retreat_pos = Position(best_retreat[0], best_retreat[1])
                move = self._get_move_towards(current, retreat_pos)
                if move:
                    return move
        
        # Fallback: try all directions and pick the safest
        # IMPORTANT: Prefer movement towards safe zone when possible
        current_y = current.y
        safe_zone_y = self.game_state.safe_zone_y
        
        # Prioritize moves that keep us in or return us to safe zone
        if current_y < safe_zone_y:
            # We're above safe zone - prefer moving DOWN
            moves = [
                ('s', Position(current.x, current.y + 1)),  # Down - FIRST
                ('a', Position(current.x - 1, current.y)),
                ('d', Position(current.x + 1, current.y)),
                ('w', Position(current.x, current.y - 1)),  # Up - LAST
            ]
        elif current_y >= self.game_state.map_height - 2:
            # We're at bottom edge - prefer moving UP or horizontal
            moves = [
                ('w', Position(current.x, current.y - 1)),  # Up - FIRST
                ('a', Position(current.x - 1, current.y)),
                ('d', Position(current.x + 1, current.y)),
                ('s', Position(current.x, current.y + 1)),  # Down - LAST
            ]
        else:
            # We're in safe zone - prefer horizontal movement
            moves = [
                ('a', Position(current.x - 1, current.y)),
                ('d', Position(current.x + 1, current.y)),
                ('s', Position(current.x, current.y + 1)),
                ('w', Position(current.x, current.y - 1)),
            ]
        
        # Sort by safety (distance from danger)
        safe_moves = []
        for cmd, pos in moves:
            if self.game_state.is_safe_position(pos):
                min_danger_dist = min(
                    (pos.distance_to(d) for d in self.game_state.danger_zones),
                    default=100
                )
                safe_moves.append((min_danger_dist, cmd))
        
        if safe_moves:
            safe_moves.sort(reverse=True)  # Furthest from danger first
            return safe_moves[0][1]
        
        # Desperate: any move that doesn't immediately kill us
        for cmd, pos in moves:
            if 0 <= pos.x < self.game_state.map_width and 0 <= pos.y < self.game_state.map_height:
                if pos not in self.game_state.mushrooms:
                    return cmd
        
        return None
    
    def _unstuck_maneuver(self, current: Position) -> Optional[str]:
        """
        Special maneuver when player is stuck (can't reach target due to obstacles)
        Strategy: Try vertical movement first, then horizontal, to navigate around obstacles
        """
        # First, try to find if there's a mushroom blocking immediate path
        blocking_mushroom = None
        
        # Check all 4 directions for mushrooms
        adjacent_mushrooms = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Up, Down, Left, Right
            check_pos = Position(current.x + dx, current.y + dy)
            if check_pos in self.game_state.mushrooms:
                adjacent_mushrooms.append((check_pos, dx, dy))
        
        # If there's a mushroom blocking, try to shoot it if we can
        for mushroom_pos, dx, dy in adjacent_mushrooms:
            # If mushroom is above us and path is clear
            if dy < 0 and mushroom_pos.x == current.x:
                logger.debug(f"Unstuck: Shooting blocking mushroom at {mushroom_pos.to_tuple()}")
                return "A"
        
        # Try vertical movement to get around obstacles
        # Prioritize moving UP first (towards action), then DOWN
        moves_priority = [
            ('w', Position(current.x, current.y - 1)),  # Up
            ('s', Position(current.x, current.y + 1)),  # Down
            ('a', Position(current.x - 1, current.y)),  # Left
            ('d', Position(current.x + 1, current.y)),  # Right
        ]
        
        # Try each move in priority order
        for cmd, pos in moves_priority:
            # Check if position is valid and safe
            if (0 <= pos.x < self.game_state.map_width and 
                0 <= pos.y < self.game_state.map_height):
                
                # Must be free of mushrooms
                if pos not in self.game_state.mushrooms:
                    # Prefer positions that aren't in immediate danger
                    if pos not in self.game_state.danger_zones:
                        logger.debug(f"Unstuck: Moving {cmd} to {pos.to_tuple()}")
                        return cmd
        
        # If all safe moves are blocked, try any move that's at least valid
        for cmd, pos in moves_priority:
            if (0 <= pos.x < self.game_state.map_width and 
                0 <= pos.y < self.game_state.map_height):
                if pos not in self.game_state.mushrooms:
                    logger.warning(f"Unstuck (desperate): Moving {cmd} to {pos.to_tuple()}")
                    return cmd
        
        # Completely stuck - try shooting to clear path
        logger.warning("Completely stuck! Shooting to clear path")
        return "A"
    
    def _get_move_towards(self, current: Position, target: Position) -> Optional[str]:
        """Get single move command towards target"""
        path = self.pathfinder.find_path(current, target)
        
        if path and len(path) > 0:
            next_pos = path[0]
            dx = next_pos.x - current.x
            dy = next_pos.y - current.y
            
            if dy < 0:
                return 'w'
            elif dy > 0:
                return 's'
            elif dx < 0:
                return 'a'
            elif dx > 0:
                return 'd'
        
        # If pathfinding failed, try direct movement
        dx = target.x - current.x
        dy = target.y - current.y
        
        # Prioritize larger movement
        if abs(dx) > abs(dy):
            # Try horizontal movement first
            if dx > 0:
                new_pos = Position(current.x + 1, current.y)
                if self.game_state.is_safe_position(new_pos):
                    return 'd'
                # Horizontal blocked, try vertical as alternative
                else:
                    if dy > 0:
                        alt_pos = Position(current.x, current.y + 1)
                        if self.game_state.is_safe_position(alt_pos):
                            logger.debug("Horizontal blocked, moving down instead")
                            return 's'
                    elif dy < 0:
                        alt_pos = Position(current.x, current.y - 1)
                        if self.game_state.is_safe_position(alt_pos):
                            logger.debug("Horizontal blocked, moving up instead")
                            return 'w'
                    # Try any vertical movement
                    for vert_cmd, vert_pos in [
                        ('w', Position(current.x, current.y - 1)),
                        ('s', Position(current.x, current.y + 1))
                    ]:
                        if self.game_state.is_safe_position(vert_pos):
                            logger.debug(f"Horizontal blocked, trying vertical: {vert_cmd}")
                            return vert_cmd
                            
            elif dx < 0:
                new_pos = Position(current.x - 1, current.y)
                if self.game_state.is_safe_position(new_pos):
                    return 'a'
                # Horizontal blocked, try vertical as alternative
                else:
                    if dy > 0:
                        alt_pos = Position(current.x, current.y + 1)
                        if self.game_state.is_safe_position(alt_pos):
                            logger.debug("Horizontal blocked, moving down instead")
                            return 's'
                    elif dy < 0:
                        alt_pos = Position(current.x, current.y - 1)
                        if self.game_state.is_safe_position(alt_pos):
                            logger.debug("Horizontal blocked, moving up instead")
                            return 'w'
                    # Try any vertical movement
                    for vert_cmd, vert_pos in [
                        ('w', Position(current.x, current.y - 1)),
                        ('s', Position(current.x, current.y + 1))
                    ]:
                        if self.game_state.is_safe_position(vert_pos):
                            logger.debug(f"Horizontal blocked, trying vertical: {vert_cmd}")
                            return vert_cmd
        
        # Try vertical movement
        if dy > 0:
            new_pos = Position(current.x, current.y + 1)
            if self.game_state.is_safe_position(new_pos):
                return 's'
        elif dy < 0:
            new_pos = Position(current.x, current.y - 1)
            if self.game_state.is_safe_position(new_pos):
                return 'w'
        
        return None
    
    def _find_safest_position_in_safe_zone(self) -> Optional[Position]:
        """Find the safest position in the bottom safe zone"""
        safest_pos = None
        max_safety = -1
        
        for x in range(self.game_state.map_width):
            for y in range(self.game_state.safe_zone_y, self.game_state.map_height):
                pos = Position(x, y)
                
                if not self.game_state.is_safe_position(pos):
                    continue
                
                # Calculate safety score
                safety_score = 0
                
                # Distance from danger
                min_danger_dist = min(
                    (pos.distance_to(d) for d in self.game_state.danger_zones),
                    default=100
                )
                safety_score += min_danger_dist * 10
                
                # Prefer center positions
                center_dist = abs(x - self.game_state.map_width // 2)
                safety_score -= center_dist
                
                if safety_score > max_safety:
                    max_safety = safety_score
                    safest_pos = pos
        
        return safest_pos
    
    def _stay_mobile(self) -> str:
        """Make small movements to stay mobile and ready"""
        current = self.game_state.bug_blaster_pos
        
        # PRIORITY: Try to shoot if we have opportunity
        if self.game_state.centipedes:
            for centipede in self.game_state.centipedes:
                for seg_x, seg_y in centipede.body:
                    if seg_x == current.x and seg_y < current.y:
                        # Check if path is clear
                        clear = True
                        for y in range(seg_y + 1, current.y):
                            if Position(current.x, y) in self.game_state.mushrooms:
                                clear = False
                                break
                        if clear:
                            logger.debug(f"Stay mobile: opportunistic shot at {seg_x},{seg_y}")
                            return 'A'
        
        # If there are centipedes, try to position better for shooting
        if self.game_state.centipedes:
            nearest = self.game_state.get_nearest_centipede_head()
            if nearest and nearest.head.y < current.y:
                # Try to align horizontally with centipede
                if abs(nearest.head.x - current.x) > 2:
                    if nearest.head.x < current.x:
                        return 'a'
                    else:
                        return 'd'
        
        # IMPORTANT: If we're above safe zone and no immediate threat, move down
        if current.y < self.game_state.safe_zone_y:
            down_pos = Position(current.x, current.y + 1)
            if self.game_state.is_safe_position(down_pos):
                logger.debug(f"Stay mobile: returning to safe zone from y={current.y}")
                return 's'
        
        # Prefer horizontal movement in safe zone, avoiding edges
        if current.x < 3:
            return 'd'
        elif current.x > self.game_state.map_width - 4:
            return 'a'
        elif current.x < self.game_state.map_width // 3:
            return 'd'
        elif current.x > 2 * self.game_state.map_width // 3:
            return 'a'
        else:
            # Small movement based on step to vary position
            if self.game_state.step % 6 < 3:
                return 'd'
            else:
                return 'a'
    
    def _smart_patrol(self) -> str:
        """Smart patrol movements - actively seek centipedes"""
        current = self.game_state.bug_blaster_pos
        
        # PRIORITY: Try to shoot if aligned with any centipede
        if self.game_state.centipedes:
            for centipede in self.game_state.centipedes:
                for seg_x, seg_y in centipede.body:
                    if seg_x == current.x and seg_y < current.y:
                        # Check if path is clear
                        clear = True
                        for y in range(seg_y + 1, current.y):
                            if Position(current.x, y) in self.game_state.mushrooms:
                                clear = False
                                break
                        if clear:
                            logger.debug(f"Smart patrol: shooting at {seg_x},{seg_y}")
                            return 'A'
        
        # If there are centipedes, try to position under them
        if self.game_state.centipedes:
            nearest = self.game_state.get_nearest_centipede_head()
            if nearest and nearest.head.y < current.y:
                # Try to move towards the centipede's x position
                if nearest.head.x < current.x - 1:
                    return 'a'
                elif nearest.head.x > current.x + 1:
                    return 'd'
                # If we're roughly aligned but ABOVE safe zone, DON'T move up more!
                # Only move up if we're IN the safe zone and want to get closer
                elif current.y > self.game_state.safe_zone_y:
                    target = Position(current.x, current.y - 1)
                    if self.game_state.is_safe_position(target):
                        return 'w'
        
        # IMPORTANT: If we're above safe zone, return to it
        if current.y < self.game_state.safe_zone_y:
            down_pos = Position(current.x, current.y + 1)
            if self.game_state.is_safe_position(down_pos):
                logger.debug(f"Smart patrol: returning to safe zone from y={current.y}")
                return 's'
        
        # Default patrol behavior - avoid edges
        if current.x <= 2:
            return 'd'
        elif current.x >= self.game_state.map_width - 3:
            return 'a'
        else:
            # Move towards center slightly
            if current.x < self.game_state.map_width // 2:
                return 'd'
            else:
                return 'a'


async def agent_loop(server_address="localhost:8000", agent_name="student"):
    """Main agent loop"""
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        # Join game
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))
        logger.info(f"Agent {agent_name} joined the game")
        
        # Initialize game state
        game_state = GameState()
        strategy_manager = StrategyManager(game_state)
        
        # Display window for human viewing (optional)
        SCREEN = pygame.display.set_mode((299, 123))
        SPRITES = pygame.image.load("data/pad.png").convert_alpha()
        SCREEN.blit(SPRITES, (0, 0))
        
        frame_count = 0
        
        while True:
            try:
                # Receive game state
                state = json.loads(await websocket.recv())
                
                # Update our game state
                game_state.update(state)
                
                # Log important events
                if frame_count % 50 == 0:
                    logger.info(f"Step: {game_state.step}, Score: {game_state.score}, "
                              f"Centipedes: {len(game_state.centipedes)}, "
                              f"Mushrooms: {len(game_state.mushrooms)}")
                
                # Decide action with error handling
                try:
                    key = strategy_manager.decide_action()
                except Exception as action_error:
                    # Log the error but don't crash - use a safe fallback
                    logger.error(f"Error deciding action: {action_error}", exc_info=True)
                    # Fallback: simple patrol movement
                    if game_state.bug_blaster_pos:
                        if game_state.bug_blaster_pos.x < game_state.map_width // 2:
                            key = 'd'
                        else:
                            key = 'a'
                    else:
                        key = ''  # No action if position unknown
                
                # Send action
                await websocket.send(json.dumps({"cmd": "key", "key": key}))
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                pygame.display.flip()
                frame_count += 1
                
            except websockets.exceptions.ConnectionClosedOK:
                logger.info("Server has cleanly disconnected us")
                return
            except Exception as e:
                logger.error(f"Error in agent loop: {e}", exc_info=True)
                break


# Run the agent
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    SERVER = os.environ.get("SERVER", "localhost")
    PORT = os.environ.get("PORT", "8000")
    NAME = os.environ.get("NAME", getpass.getuser())
    
    try:
        loop.run_until_complete(agent_loop(f"{SERVER}:{PORT}", NAME))
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)