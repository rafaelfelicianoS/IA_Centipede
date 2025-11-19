"""
Centipede AI Agent
Authors: [Your names and mec numbers here]

Architecture:
- Perception: Read game state
- Prediction: Future centipede movement simulation
- Hazard mapping: Danger zone calculation
- Strategy selection: Aggressive/Defensive/Clearing
- Movement + shooting logic with safety evaluation
- Emergency overrides for critical situations

CRITICAL: CENTIPEDE HEAD DEFINITION
====================================
Throughout this code, the centipede HEAD is ALWAYS body[-1] (last element).
The TAIL is body[0] (first element).
This must be consistent everywhere to avoid oscillation bugs (adadad loops).

CAMPING MODE (v3):
==================
When targeting stuck centipedes, the agent enters "camping mode":
- Stay stationary and shoot continuously when cooldown allows
- No recentering, no lateral movements, no ADA/ADADA cycles
- Maintains defensive awareness for threats

Camping mode threats (will exit camping):
1. Free centipedes approaching in same/adjacent column (within 5 rows)
2. Free centipedes within 3 rows and descending
3. Stuck centipede target becomes unstuck
4. Line of fire blocked by >3 mushrooms (needs repositioning)

Mid game vs Late game behavior:
- Mid game: Camp only if aligned, no nearby threats (8+ units), clearly safe
- Late game: Auto-enter camping when aligned with stuck target (priority target)

Camping mode exits only for logical reasons, never by timeout.
Self-stuck detection is disabled during camping to avoid false positives.

LATE GAME CLEARING LOGIC (v2):
==============================
In late game mode, the agent is authorized to clear mushrooms ONLY when:
1. The current target is a stuck centipede
2. Mushrooms are blocking the direct shot path (same column, between blaster and centipede head)
3. Maximum 3 mushrooms in path (to avoid endless clearing)
4. It's safe to shoot (no immediate death risk)

Priority in late game:
1. Kill centipede (always highest priority when path is clear)
2. Clear mushrooms blocking path to stuck centipede (limited clearing)
3. Move to different position if too many mushrooms or target not stuck

The agent NEVER:
- Shoots mushrooms randomly in late game
- Prioritizes mushroom clearing over centipede killing when line is clear
- Enters "panic mode" when well-positioned in safe zone against stuck centipedes

ADADADA Loop Prevention:
- When horizontal oscillation detected (alternating a/d), triggers shooting to clear path
- Applies when targeting stuck centipedes with mushrooms blocking

RETURN TO SAFE ZONE (v2):
=========================
When the agent is returning to the safe zone, a special priority system is used
to eliminate wswsws oscillation patterns:

Priority order: s > a/d > w
1. 's' (down) - HIGHEST PRIORITY when safe
   - Gets +500 bonus for descending toward safe zone
   - Minimal distance penalty consideration
2. 'a'/'d' (lateral) - SECONDARY PRIORITY
   - Gets +200 bonus for repositioning
   - Used to escape tight spots or reposition when 's' is blocked
3. 'w' (up) - LAST RESORT ONLY
   - Gets -100 penalty to discourage upward movement
   - Only chosen when s, a, d are all dangerous or blocked

Self-stuck vertical bonus is DISABLED in this mode to prevent 'w' from gaining
artificial priority that causes wswsws patterns.
"""

import asyncio
import getpass
import json
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
import math
import logging

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CentipedeAgent')

# Game constants
DIRECTIONS = {
    'w': (0, -1),  # UP
    's': (0, 1),   # DOWN
    'a': (-1, 0),  # LEFT
    'd': (1, 0),   # RIGHT
}

# Late game threshold - when mushroom count >= this, enter late game mode
LATE_GAME_MUSHROOM_THRESHOLD = 160

class Position:
    """Represents a position on the game grid"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return isinstance(other, Position) and self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __repr__(self):
        return f"Pos({self.x},{self.y})"
    
    def to_tuple(self):
        return (self.x, self.y)
    
    def manhattan_distance(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class CentipedeAgent:
    """
    Main agent class implementing multi-layered decision system
    """
    
    def __init__(self):
        # Core state
        self.game_state = None
        self.map_info = None
        self.map_size = None
        
        # Tracking
        self.shot_cooldown = 0
        self.centipede_tracking = defaultdict(lambda: deque(maxlen=10))
        self.predicted_positions = {}
        self.danger_zones = set()
        
        # Centipede stuck detection
        self.centipede_position_history = defaultdict(lambda: deque(maxlen=15))
        self.stuck_centipedes = set()
        
        # Bug blaster self-stuck detection
        self.blaster_position_history = deque(maxlen=30)  # Last 30 positions
        self.blaster_column_history = deque(maxlen=30)  # Last 30 X coordinates
        self.self_stuck_detected = False
        self.self_stuck_cooldown = 0  # Frames until next self-stuck check
        self.last_target_name = None
        self.frames_on_same_target = 0
        
        # Late game state
        self.late_game = False
        self.focus_target_name = None
        
        # Camping mode state
        self.camping_mode = False
        self.camping_target_name = None
        self.camping_start_frame = 0
        
        # Strategy
        self.current_strategy = "aggressive"
        self.target_column = None
        self.safe_zone_start = None
        
        # Frame tracking
        self.frame_count = 0
        
        # Debug
        self.last_action = ""
        self.debug_info = {}
        
    def update_state(self, state: dict):
        """Main state update - the brain heartbeat"""
        self.game_state = state
        self.frame_count += 1
        
        # Initialize map info on first frame
        if self.map_info is None:
            self.map_info = {
                'size': state.get('size', (40, 24)),
                'map': state.get('map', [])
            }
            self.map_size = self.map_info['size']
            self.safe_zone_start = self.map_size[1] - 5  # Bottom 5 rows
        
        # Calculate late game status based on mushroom count
        mushroom_count = len(state.get('mushrooms', []))
        was_late_game = self.late_game
        self.late_game = mushroom_count >= LATE_GAME_MUSHROOM_THRESHOLD
        
        if self.late_game and not was_late_game:
            logger.info(f"🎯 ENTERING LATE GAME MODE - {mushroom_count} mushrooms (threshold: {LATE_GAME_MUSHROOM_THRESHOLD})")
        elif not self.late_game and was_late_game:
            logger.info(f"✓ Exiting late game mode - {mushroom_count} mushrooms")
            self.focus_target_name = None  # Reset focus target
            # Reset camping mode when exiting late game
            if self.camping_mode:
                logger.info("🏕️ Exiting camping mode (late game ended)")
                self.camping_mode = False
                self.camping_target_name = None
        
        # Reduce cooldown
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1
        
        # Update predictions
        self.update_centipede_predictions()
        
        # Rebuild danger zones
        self.update_danger_zones()
        
        # Re-evaluate strategy
        self.evaluate_strategy()
    
    def update_centipede_predictions(self):
        """Predict future centipede positions (5 steps ahead)"""
        self.predicted_positions = {}
        
        centipedes = self.game_state.get('centipedes', [])
        mushrooms = self.get_mushroom_positions()
        
        for centipede in centipedes:
            name = centipede['name']
            body = centipede['body']
            direction = centipede['direction']  # 0=N, 1=E, 2=S, 3=W
            
            if not body:
                continue
            
            # Track history
            head = body[-1]  # Head is last element
            self.centipede_tracking[name].append(head)
            
            # Track position history for stuck detection
            head_tuple = tuple(head)
            self.centipede_position_history[name].append(head_tuple)
            
            # Simulate 5 steps ahead
            predictions = []
            sim_head = list(head)
            sim_dir = direction
            move_dir = 1  # 1 = down, -1 = up
            
            for step in range(5):
                # Determine horizontal direction
                if sim_dir == 1:  # EAST
                    next_x = sim_head[0] + 1
                elif sim_dir == 3:  # WEST
                    next_x = sim_head[0] - 1
                else:
                    next_x = sim_head[0]
                
                next_pos = [next_x, sim_head[1]]
                
                # Check collision with mushroom or wall
                hit_obstacle = False
                if next_x < 0 or next_x >= self.map_size[0]:
                    hit_obstacle = True
                elif tuple(next_pos) in mushrooms:
                    hit_obstacle = True
                
                if hit_obstacle:
                    # Move down/up and reverse direction
                    if sim_head[1] == 0:
                        move_dir = 1
                    elif sim_head[1] >= self.map_size[1] - 1:
                        move_dir = -1
                    
                    next_pos = [sim_head[0], sim_head[1] + move_dir]
                    
                    # Reverse horizontal direction
                    if sim_dir == 1:  # EAST -> WEST
                        sim_dir = 3
                    elif sim_dir == 3:  # WEST -> EAST
                        sim_dir = 1
                
                sim_head = next_pos
                predictions.append(tuple(sim_head))
            
            self.predicted_positions[name] = predictions
    
    def detect_stuck_centipedes(self) -> List[str]:
        """
        Detect centipedes that are stuck (not moving for several frames)
        Returns list of stuck centipede names
        """
        stuck = []
        
        for name, history in self.centipede_position_history.items():
            if len(history) >= 10:
                # Check if centipede hasn't moved in last 10 frames
                recent_positions = list(history)[-10:]
                unique_positions = set(recent_positions)
                
                # If only 1-2 unique positions in 10 frames, it's stuck
                if len(unique_positions) <= 2:
                    stuck.append(name)
                    if name not in self.stuck_centipedes:
                        logger.warning(f"Centipede {name} detected as STUCK at position {recent_positions[-1]}")
                        self.stuck_centipedes.add(name)
                else:
                    # Remove from stuck set if it started moving
                    if name in self.stuck_centipedes:
                        logger.info(f"Centipede {name} is now moving again")
                        self.stuck_centipedes.discard(name)
        
        return stuck
    
    def detect_self_stuck(self) -> bool:
        """
        Detect if bug blaster is stuck in extreme/rare patterns
        Very strict conditions to avoid interfering with normal gameplay
        
        Patterns detected:
        1. Alternating between two columns repeatedly (adadadadad pattern)
        2. All centipedes stuck but blaster not positioned under any of them
        
        Returns True if self-stuck detected
        
        NOTE: This detection is DISABLED when camping_mode is active to avoid
        false positives when the agent is deliberately stationary.
        """
        # CAMPING MODE OVERRIDE: Skip self-stuck detection when camping
        # The agent is deliberately stationary, so it's not stuck
        if self.camping_mode:
            return False
        
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return False
        
        my_pos = Position(*bug_blaster['pos'])
        
        # Track current position
        self.blaster_position_history.append(my_pos.to_tuple())
        self.blaster_column_history.append(my_pos.x)
        
        # Don't check too frequently
        if self.self_stuck_cooldown > 0:
            self.self_stuck_cooldown -= 1
            return self.self_stuck_detected
        
        # Need enough history
        if len(self.blaster_column_history) < 20:
            return False
        
        # Check if there are alive centipedes
        centipedes = self.game_state.get('centipedes', [])
        if not centipedes:
            return False
        
        # Pattern 1: Alternating between two columns (adadadadad)
        recent_columns = list(self.blaster_column_history)[-20:]
        unique_columns = set(recent_columns)
        
        if len(unique_columns) == 2:
            # Check if alternating pattern
            col_a, col_b = sorted(unique_columns)
            alternations = 0
            for i in range(1, len(recent_columns)):
                if recent_columns[i] != recent_columns[i-1]:
                    alternations += 1
            
            # If alternating frequently (>12 changes in 20 frames)
            if alternations > 12:
                if not self.self_stuck_detected:
                    logger.warning(f"⚠️ SELF-STUCK DETECTED: Alternating between columns {col_a} and {col_b} ({alternations} changes in 20 frames)")
                self.self_stuck_detected = True
                self.self_stuck_cooldown = 30  # Check again in 30 frames
                return True
        
        # Pattern 2: All centipedes stuck but blaster not under any
        # First, check if confined to small column range with stuck centipedes (Pattern 1.5)
        if len(unique_columns) <= 3:
            num_stuck = sum(1 for c in centipedes if c['name'] in self.stuck_centipedes)
            if num_stuck >= len(centipedes) * 0.5:  # At least half are stuck
                if not self.self_stuck_detected:
                    col_range = max(unique_columns) - min(unique_columns) if len(unique_columns) > 1 else 0
                    logger.warning(f"⚠️ SELF-STUCK DETECTED: Confined to {len(unique_columns)} columns (range: {col_range}) with {num_stuck}/{len(centipedes)} stuck centipedes")
                self.self_stuck_detected = True
                self.self_stuck_cooldown = 30
                return True
        
        # Pattern 2: All centipedes stuck but blaster not under any
        all_stuck = len(centipedes) > 0 and all(c['name'] in self.stuck_centipedes for c in centipedes)
        
        if all_stuck and len(centipedes) >= 2:
            # Check if blaster is aligned with any stuck centipede
            aligned_with_any = False
            for centipede in centipedes:
                if centipede['body']:
                    head_x = centipede['body'][-1][0]  # Head is last element
                    if abs(my_pos.x - head_x) <= 1:  # Within 1 column
                        aligned_with_any = True
                        break
            
            # Check if we've been far from all stuck centipedes for a while
            if not aligned_with_any:
                # Count frames where we were far from all stuck centipedes
                recent_positions = list(self.blaster_position_history)[-15:]
                frames_far_from_all = 0
                
                for pos in recent_positions:
                    far_from_all = True
                    for centipede in centipedes:
                        if centipede['body']:
                            head_x = centipede['body'][-1][0]  # Head is last element
                            if abs(pos[0] - head_x) <= 2:
                                far_from_all = False
                                break
                    if far_from_all:
                        frames_far_from_all += 1
                
                # If far from all stuck centipedes for > 12 frames
                if frames_far_from_all > 12:
                    if not self.self_stuck_detected:
                        logger.warning(f"⚠️ SELF-STUCK DETECTED: All {len(centipedes)} centipedes stuck but blaster not aligned with any for {frames_far_from_all} frames")
                    self.self_stuck_detected = True
                    self.self_stuck_cooldown = 30
                    return True
        
        # Clear self-stuck if patterns not detected
        if self.self_stuck_detected:
            logger.info("✓ Self-stuck condition cleared")
            self.self_stuck_detected = False
        
        return False
    
    def update_danger_zones(self):
        """Create heat map of dangerous positions"""
        self.danger_zones = set()
        
        centipedes = self.game_state.get('centipedes', [])
        
        for centipede in centipedes:
            body = centipede['body']
            name = centipede['name']
            
            # Add zones around all body segments
            for segment in body:
                self.danger_zones.add(tuple(segment))
                # Add adjacent cells
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    adj = (segment[0] + dx, segment[1] + dy)
                    if 0 <= adj[0] < self.map_size[0] and 0 <= adj[1] < self.map_size[1]:
                        self.danger_zones.add(adj)
            
            # Add zones around predicted positions (next 1-3 steps)
            predictions = self.predicted_positions.get(name, [])
            for pred in predictions[:3]:
                self.danger_zones.add(pred)
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    adj = (pred[0] + dx, pred[1] + dy)
                    if 0 <= adj[0] < self.map_size[0] and 0 <= adj[1] < self.map_size[1]:
                        self.danger_zones.add(adj)
    
    def will_be_hit_soon(self, pos: Position, horizon: int = 2) -> bool:
        """
        Predictive death check: Will any centipede HEAD reach this position 
        within the next 'horizon' steps?
        
        This is critical because the game moves centipedes BEFORE processing
        bug blaster movement. So even if we "decide" to flee, if a centipede
        head will be at our target position in the next frame, we die.
        
        Args:
            pos: Position to check (where we would be after moving)
            horizon: Number of future steps to check (1-2 recommended)
        
        Returns:
            True if position will be hit by a centipede head soon (DANGEROUS)
            False if position appears safe from predicted head positions
        """
        pos_tuple = pos.to_tuple()
        
        # Check all centipede predictions
        for name, predictions in self.predicted_positions.items():
            # Check up to horizon steps ahead
            for step_idx in range(min(horizon, len(predictions))):
                predicted_head = predictions[step_idx]
                if predicted_head == pos_tuple:
                    logger.debug(f"⚠️ PREDICTED DEATH: Centipede '{name}' head will be at {pos_tuple} in step {step_idx + 1}")
                    return True
        
        # Also check current head positions (step 0)
        centipedes = self.game_state.get('centipedes', [])
        for centipede in centipedes:
            body = centipede['body']
            if body:
                current_head = tuple(body[-1])  # Head is last element
                if current_head == pos_tuple:
                    logger.debug(f"⚠️ IMMEDIATE DEATH: Centipede head currently at {pos_tuple}")
                    return True
        
        return False
    
    def get_safest_action_with_prediction(self, candidate_actions: list, my_pos: Position) -> Optional[str]:
        """
        Given a list of candidate actions, filter out those that lead to predicted death
        and return the safest one based on distance from threats.
        
        Args:
            candidate_actions: List of actions to evaluate ['a', 'd', 'w', 's', '']
            my_pos: Current position
        
        Returns:
            Safest action, or None if all are dangerous
        """
        mushrooms = self.get_mushroom_positions()
        centipedes = self.game_state.get('centipedes', [])
        
        safe_actions = []
        risky_actions = []  # Actions that might be hit but are better than certain death
        
        for action in candidate_actions:
            # Calculate new position
            if action == '':
                new_pos = my_pos
            else:
                if action not in DIRECTIONS:
                    continue
                dx, dy = DIRECTIONS[action]
                new_pos = Position(my_pos.x + dx, my_pos.y + dy)
            
            # Check if move is valid
            if new_pos.x < 0 or new_pos.x >= self.map_size[0]:
                continue
            if new_pos.y < 0 or new_pos.y >= self.map_size[1]:
                continue
            if new_pos.to_tuple() in mushrooms:
                continue
            
            # Check if move leads to immediate centipede collision
            immediate_collision = False
            for centipede in centipedes:
                if new_pos.to_tuple() in [tuple(seg) for seg in centipede['body']]:
                    immediate_collision = True
                    break
            
            if immediate_collision:
                continue
            
            # Check predicted death
            if self.will_be_hit_soon(new_pos, horizon=2):
                # This action leads to predicted death, but might be better than nothing
                min_dist = float('inf')
                for centipede in centipedes:
                    for segment in centipede['body']:
                        dist = new_pos.manhattan_distance(Position(*segment))
                        min_dist = min(min_dist, dist)
                risky_actions.append((action, min_dist))
            else:
                # Safe from predicted death - calculate distance score
                min_dist = float('inf')
                for centipede in centipedes:
                    for segment in centipede['body']:
                        dist = new_pos.manhattan_distance(Position(*segment))
                        min_dist = min(min_dist, dist)
                safe_actions.append((action, min_dist))
        
        # Prefer safe actions (sorted by distance, farther is better)
        if safe_actions:
            safe_actions.sort(key=lambda x: x[1], reverse=True)
            return safe_actions[0][0]
        
        # If no safe actions, pick the least risky
        if risky_actions:
            risky_actions.sort(key=lambda x: x[1], reverse=True)
            logger.warning(f"⚠️ No safe actions available, choosing least risky: {risky_actions[0][0]}")
            return risky_actions[0][0]
        
        return None
    
    def evaluate_strategy(self):
        """Choose between AGGRESSIVE, DEFENSIVE, or CLEARING"""
        bug_blaster = self.game_state.get('bug_blaster', {})
        
        # Validate bug_blaster has position
        if not bug_blaster or 'pos' not in bug_blaster:
            self.current_strategy = "aggressive"
            return
        
        my_pos = Position(*bug_blaster['pos'])
        
        centipedes = self.game_state.get('centipedes', [])
        mushroom_count = len(self.game_state.get('mushrooms', []))
        
        # Detect stuck centipedes
        stuck_centipedes = self.detect_stuck_centipedes()
        
        old_strategy = self.current_strategy
        
        # PRIORITY: If there are stuck centipedes, ALWAYS go aggressive to kill them
        # This takes priority over defensive mode to avoid "panic and flee" behavior
        # when we're well-positioned to shoot stuck targets
        if stuck_centipedes:
            # Check if we're in safe zone - if so, stay aggressive even with close threats
            in_safe_zone = my_pos.y >= self.safe_zone_start
            
            # Check if the closest threat is a stuck centipede
            closest_is_stuck = False
            min_distance = float('inf')
            closest_centipede_name = None
            
            for centipede in centipedes:
                for segment in centipede['body']:
                    seg_pos = Position(*segment)
                    dist = my_pos.manhattan_distance(seg_pos)
                    if dist < min_distance:
                        min_distance = dist
                        closest_centipede_name = centipede['name']
            
            if closest_centipede_name and closest_centipede_name in self.stuck_centipedes:
                closest_is_stuck = True
            
            # Don't switch to defensive if:
            # 1. We're in safe zone AND closest threat is stuck
            # 2. OR distance is >= 3 (not immediately dangerous)
            if in_safe_zone and closest_is_stuck:
                self.current_strategy = "aggressive"
                if old_strategy != "aggressive":
                    logger.info(f"Staying AGGRESSIVE - in safe zone with stuck target '{closest_centipede_name}' (dist: {min_distance})")
                return
            elif min_distance >= 3:
                self.current_strategy = "aggressive"
                if old_strategy != "aggressive":
                    logger.info(f"Switching to AGGRESSIVE strategy - {len(stuck_centipedes)} stuck centipede(s), distance safe ({min_distance})")
                return
            else:
                # Immediate threat from non-stuck centipede - go defensive but stay aware
                pass  # Fall through to defensive check below
        
        # DEFENSIVE: Any centipede close (< 4 distance) and below/near our row
        min_distance = float('inf')
        threat_below = False
        
        for centipede in centipedes:
            for segment in centipede['body']:



                seg_pos = Position(*segment)
                dist = my_pos.manhattan_distance(seg_pos)
                if dist < min_distance:
                    min_distance = dist
                if seg_pos.y >= my_pos.y - 2:  # Near or below us
                    threat_below = True
        
        if min_distance < 4 and threat_below:
            # Additional check: don't go defensive if all close threats are stuck centipedes
            # and we're in a position to shoot them
            all_close_threats_stuck = True
            for centipede in centipedes:
                for segment in centipede['body']:
                    seg_pos = Position(*segment)
                    dist = my_pos.manhattan_distance(seg_pos)
                    if dist < 4 and centipede['name'] not in self.stuck_centipedes:
                        all_close_threats_stuck = False
                        break
                if not all_close_threats_stuck:
                    break
            
            if all_close_threats_stuck and stuck_centipedes:
                # All close threats are stuck - stay aggressive to kill them
                self.current_strategy = "aggressive"
                if old_strategy != "aggressive":
                    logger.info(f"Staying AGGRESSIVE - all close threats ({min_distance}) are stuck centipedes")
                return
            
            self.current_strategy = "defensive"
            if old_strategy != "defensive":
                logger.info(f"Switching to DEFENSIVE strategy - centipede at distance {min_distance:.1f}")
            return
        
        # CLEARING: Too many mushrooms (> 150) AND no stuck centipedes
        # BUT: In late game, NEVER enter clearing mode - stay aggressive to hunt centipedes
        if mushroom_count > 150 and not self.late_game:
            self.current_strategy = "clearing"
            if old_strategy != "clearing":
                logger.info(f"Switching to CLEARING strategy - {mushroom_count} mushrooms (no stuck centipedes)")
            return
        
        # AGGRESSIVE: Default - hunt and kill
        # In late game, always aggressive (focused hunting)
        self.current_strategy = "aggressive"
        if old_strategy != "aggressive":
            if self.late_game:
                logger.info("Switching to AGGRESSIVE strategy (late game mode)")
            else:
                logger.info("Switching to AGGRESSIVE strategy")
    
    def find_best_target(self) -> Optional[Tuple[str, Tuple[int, int], float]]:
        """
        Find the best centipede segment to target with balanced scoring
        
        SCORING LOGIC:
        ===============
        1. STUCK BONUS (+200): Relevant but not dominant
           - Stuck centipedes are easier to hit but distance still matters
        
        2. DISTANCE PENALTY (-2 per Manhattan distance unit):
           - Penalizes targets far from blaster
           - Ensures we don't chase distant targets endlessly
           - Manhattan distance = |dx| + |dy|
        
        3. LATE GAME LOW-Y BONUS (in late game only, +0.5 per Y coordinate):
           - Moderate bonus for centipedes lower in map
           - Encourages killing accessible stuck centipedes in bottom area
           - Not strong enough to override distance penalty
        
        4. HEIGHT SCORE (+10 per row from bottom):
           - Rewards killing centipedes higher up (more game points)
        
        5. HEAD BONUS (+50):
           - Prioritize head/tail shots
        
        6. COLUMN ALIGNMENT BONUS (+100):
           - Major bonus when already aligned
           - +150 extra if path is clear
        
        7. SELF-STUCK MITIGATION:
           - If self-stuck detected and this is current target: -100 penalty
           - Encourages switching to other targets
        
        Returns: (centipede_name, segment_position, score) or None
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        
        # Validate bug_blaster has position
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        
        my_pos = Position(*bug_blaster['pos'])
        
        centipedes = self.game_state.get('centipedes', [])
        mushrooms = self.get_mushroom_positions()
        
        # Detect self-stuck pattern
        self_stuck = self.detect_self_stuck()
        
        targets = []
        
        for centipede in centipedes:
            name = centipede['name']
            body = centipede['body']
            
            # Check if this centipede is stuck
            is_stuck = name in self.stuck_centipedes
            
            # Track if this is current target
            is_current_target = (name == self.last_target_name)
            
            for idx, segment in enumerate(body):
                seg_pos = Position(*segment)
                
                # Only target segments above us
                if seg_pos.y >= my_pos.y:
                    continue
                
                score = 0.0
                
                # ===== 1. STUCK BONUS =====
                # Moderate bonus - relevant but can be overridden by distance
                if is_stuck:
                    score += 200
                
                # ===== 2. DISTANCE PENALTY =====
                # Manhattan distance between blaster and segment
                distance = abs(seg_pos.x - my_pos.x) + abs(seg_pos.y - my_pos.y)
                score -= distance * 2  # Penalty: -2 per distance unit
                
                # ===== 3. LATE GAME LOW-Y BONUS =====
                # In late game, prefer centipedes lower in map (higher Y coordinate)
                # This is moderate - helps choose between stuck centipedes at different heights
                if self.late_game:
                    score += seg_pos.y * 0.5  # Small bonus per Y coordinate
                
                # ===== 4. HEIGHT SCORE =====
                # Higher on screen = more points in game
                score += (self.map_size[1] - seg_pos.y) * 10
                
                # ===== 5. HEAD BONUS =====
                # Head shots split centipede efficiently
                if idx == len(body) - 1:  # Last element is head
                    score += 50
                
                # ===== 6. COLUMN ALIGNMENT BONUS =====
                if seg_pos.x == my_pos.x:
                    score += 100
                    
                    # Path clear bonus
                    path_clear = True
                    for y in range(seg_pos.y + 1, my_pos.y):
                        if (seg_pos.x, y) in mushrooms:
                            path_clear = False
                            break
                    if path_clear:
                        score += 150
                
                # ===== 7. SELF-STUCK MITIGATION =====
                # If we're stuck on this target, penalize it to encourage switching
                if self_stuck and is_current_target:
                    score -= 100
                    logger.debug(f"Self-stuck mitigation: reducing score for current target '{name}' by -100")
                
                # ===== 8. PROXIMITY PENALTY =====
                # Don't get too close (dangerous)
                if distance < 3:
                    score -= 100
                
                # ===== 9. PREDICTION BONUS =====
                predictions = self.predicted_positions.get(name, [])
                if tuple(segment) in predictions[:2]:
                    score += 30
                
                targets.append((name, tuple(segment), score))
        
        if not targets:
            return None
        
        # Sort by score (highest first)
        targets.sort(key=lambda x: x[2], reverse=True)
        
        best_target = targets[0]
        best_name, best_pos, best_score = best_target
        
        # Log target changes with scores
        if best_name != self.last_target_name:
            if self.last_target_name:
                # Find old target score for comparison
                old_score = next((s for n, p, s in targets if n == self.last_target_name), None)
                if old_score is not None:
                    logger.info(f"🎯 Target changed: '{self.last_target_name}' (score: {old_score:.1f}) → '{best_name}' (score: {best_score:.1f}) at {best_pos}")
                else:
                    logger.info(f"🎯 New target: '{best_name}' (score: {best_score:.1f}) at {best_pos}")
            else:
                logger.info(f"🎯 Initial target: '{best_name}' (score: {best_score:.1f}) at {best_pos}")
            
            self.last_target_name = best_name
            self.frames_on_same_target = 0
        else:
            self.frames_on_same_target += 1
        
        # Log if targeting stuck centipede
        if best_name in self.stuck_centipedes:
            distance = abs(best_pos[0] - my_pos.x) + abs(best_pos[1] - my_pos.y)
            logger.debug(f"Targeting STUCK centipede '{best_name}' at {best_pos}, distance: {distance}, score: {best_score:.1f}")
        
        return best_target
    
    def find_safe_move(self, preferred_direction: Optional[str] = None, returning_to_safe_zone: bool = False) -> str:
        """
        Find the safest move by scoring all possible actions
        
        Args:
            preferred_direction: Optional direction hint ('w', 'a', 's', 'd')
            returning_to_safe_zone: When True, uses special priority system:
                                    Priority: s > a/d > w
                                    - 's' (down) is always preferred when safe
                                    - 'a'/'d' (lateral) are secondary options
                                    - 'w' (up) is last resort only
                                    Self-stuck vertical bonus is disabled in this mode
                                    to prevent wswsws oscillation patterns.
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        
        # Validate bug_blaster has position
        if not bug_blaster or 'pos' not in bug_blaster:
            return ''
        
        my_pos = Position(*bug_blaster['pos'])
        
        mushrooms = self.get_mushroom_positions()
        centipedes = self.game_state.get('centipedes', [])
        
        # Check if we're targeting a stuck centipede
        targeting_stuck = False
        target_stuck_pos = None
        if self.last_target_name and self.last_target_name in self.stuck_centipedes:
            targeting_stuck = True
            # Find the stuck centipede head position (head is last element)
            for c in centipedes:
                if c['name'] == self.last_target_name and c['body']:
                    target_stuck_pos = Position(*c['body'][-1])  # Head is last element
                    break
        
        best_action = ""
        best_score = -float('inf')
        
        # Evaluate all possible moves (including staying still)
        for action in ['w', 'a', 's', 'd', '']:
            if action == '':
                new_pos = my_pos
            else:
                dx, dy = DIRECTIONS[action]
                new_pos = Position(my_pos.x + dx, my_pos.y + dy)
            
            score = 0
            
            # Valid on map?
            if new_pos.x < 0 or new_pos.x >= self.map_size[0]:
                score -= 1000
                continue
            if new_pos.y < 0 or new_pos.y >= self.map_size[1]:
                score -= 1000
                continue
            
            # Mushroom collision?
            if new_pos.to_tuple() in mushrooms:
                # Normal penalty: -500 (hard block)
                # BUT: if self-stuck AND targeting stuck centipede AND mushroom is on path to target
                # Reduce penalty to allow considering controlled risk
                base_penalty = -500
                
                if self.self_stuck_detected and targeting_stuck and target_stuck_pos:
                    # Check if this mushroom is between us and the target
                    on_path_to_target = False
                    if action in ['a', 'd']:  # Horizontal movement
                        # Mushroom is on path if it's between our X and target X
                        if (my_pos.x < new_pos.x <= target_stuck_pos.x) or \
                           (target_stuck_pos.x <= new_pos.x < my_pos.x):
                            on_path_to_target = True
                    
                    if on_path_to_target:
                        # Still discourage, but less so - we might want to shoot through it
                        score += base_penalty * 0.6  # -300 instead of -500
                        logger.debug(f"Self-stuck mitigation: reduced mushroom penalty for {action} (on path to stuck target)")
                    else:
                        score += base_penalty
                else:
                    score += base_penalty
                continue
            
            # Danger zone?
            # EXCEPTION: When returning to safe zone, allow 's' and 'a'/'d' to enter danger zones
            # since we explicitly want to descend/reposition even near centipedes (but not collide)
            if new_pos.to_tuple() in self.danger_zones:
                if returning_to_safe_zone and action in ['s', 'a', 'd']:
                    # Check for immediate collision only (not danger zone)
                    immediate_collision = False
                    for centipede in centipedes:
                        if new_pos.to_tuple() in [tuple(seg) for seg in centipede['body']]:
                            immediate_collision = True
                            break
                    
                    if immediate_collision:
                        score -= 1000  # Lethal, avoid completely
                        logger.debug(f"Return to safe zone: '{action}' would collide with centipede, blocked")
                        continue
                    else:
                        # In danger zone but not immediate death - small penalty only
                        score -= 50
                        logger.debug(f"Return to safe zone: '{action}' enters danger zone but not lethal")
                else:
                    score -= 300
            
            # CRITICAL: Predictive death check
            # If moving here means a centipede head will hit us in the next 1-2 steps,
            # apply massive penalty. This is the main protection against "decided to flee
            # but died anyway" scenarios where the game moves centipedes first.
            if self.will_be_hit_soon(new_pos, horizon=2):
                score -= 5000  # Massive penalty - almost always avoid
                logger.debug(f"⚠️ Predicted death penalty for action '{action}' to {new_pos.to_tuple()}")
            
            # Distance to nearest centipede (bigger = better)
            min_centipede_dist = float('inf')
            for centipede in centipedes:
                for segment in centipede['body']:
                    seg_pos = Position(*segment)
                    dist = new_pos.manhattan_distance(seg_pos)
                    min_centipede_dist = min(min_centipede_dist, dist)
            
            # RETURN TO SAFE ZONE: Special priority system
            # Priority order: s > a/d > w
            # This eliminates wswsws oscillation patterns
            if returning_to_safe_zone:
                if action == 's':
                    # 's' (down) - HIGHEST PRIORITY when safe
                    # Minimal distance consideration - only care about immediate threats
                    if min_centipede_dist < 2:
                        score += min_centipede_dist * 5
                    else:
                        score += 20  # Small base, don't let distance dominate
                    
                    # Massive bonus for descending toward safe zone
                    if new_pos.y > my_pos.y or new_pos.y >= self.safe_zone_start:
                        score += 500
                        logger.debug(f"Return to safe zone: 's' gets priority bonus (y={my_pos.y} -> {new_pos.y})")
                
                elif action in ['a', 'd']:
                    # 'a'/'d' (lateral) - SECONDARY PRIORITY
                    # Used to escape tight spots or reposition when 's' is blocked
                    score += min_centipede_dist * 15  # Moderate distance consideration
                    score += 200  # Secondary priority bonus
                    logger.debug(f"Return to safe zone: '{action}' gets secondary priority bonus")
                
                elif action == 'w':
                    # 'w' (up) - LAST RESORT ONLY
                    # Only use when s, a, d are all dangerous or blocked
                    score += min_centipede_dist * 20  # Normal distance scoring
                    score -= 100  # Penalty to make it last resort
                    logger.debug(f"Return to safe zone: 'w' penalized as last resort")
                
                else:
                    # Staying still - moderate priority
                    score += min_centipede_dist * 10
            else:
                # Normal gameplay - standard distance scoring
                score += min_centipede_dist * 20
            
            # Stay in safe zone bonus
            if new_pos.y >= self.safe_zone_start:
                score += 50
            
            # CENTERING DISABLED:
            # Antes: penalizávamos a distância ao centro do mapa para puxar o agente
            # para o meio. Isso está agora comentado para não interferir com o foco
            # em seguir/matar cobras específicas.
            #
            # center_x = self.map_size[0] // 2
            # distance_to_center = abs(new_pos.x - center_x)
            # score -= distance_to_center * 2
            
            # Prefer requested direction
            if preferred_direction and action == preferred_direction:
                score += 100
            
            # SELF-STUCK MITIGATION: Bonus for vertical movement
            # When stuck horizontally, vertical movement helps break the pattern
            # BUT: Disabled when returning to safe zone to prevent wswsws oscillation
            if self.self_stuck_detected and action in ['w', 's'] and not returning_to_safe_zone:
                # Only give bonus if vertical move is reasonably safe
                if new_pos.to_tuple() not in self.danger_zones:
                    score += 80
                    logger.debug(f"Self-stuck mitigation: vertical movement bonus for {action}")
            
            # SELF-STUCK MITIGATION: Bonus for moving toward stuck target
            if self.self_stuck_detected and targeting_stuck and target_stuck_pos:
                # Reward moving closer to the stuck target
                current_dist = my_pos.manhattan_distance(target_stuck_pos)
                new_dist = new_pos.manhattan_distance(target_stuck_pos)
                if new_dist < current_dist:
                    score += 50
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def evade_bottom_row_snake(self) -> Optional[str]:
        """
        Special case: Avoid dying when cornered on last row
        Only active on last row
        
        CRITICAL IMPROVEMENT: Uses prediction-based safety to avoid "decided to flee
        but died anyway" scenarios where the game moves centipedes before the blaster.
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        
        # Validate bug_blaster has position
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        
        my_pos = Position(*bug_blaster['pos'])
        
        # Only on last row
        if my_pos.y != self.map_size[1] - 1:
            return None
        
        centipedes = self.game_state.get('centipedes', [])
        mushrooms = self.get_mushroom_positions()
        
        # Detect centipede within 4 cells horizontally
        threats = []
        for centipede in centipedes:
            for segment in centipede['body']:
                if segment[1] == my_pos.y:  # Same row
                    if abs(segment[0] - my_pos.x) <= 4:
                        threats.append(segment[0])
        
        if not threats:
            return None
        
        # Detect "sandwiched" scenario
        left_threat = any(x < my_pos.x for x in threats)
        right_threat = any(x > my_pos.x for x in threats)
        sandwiched = left_threat and right_threat
        
        # Build candidate actions based on threat position
        candidate_actions = []
        
        # Try UP first (most desirable escape from bottom row)
        up_pos = Position(my_pos.x, my_pos.y - 1)
        up_blocked = up_pos.to_tuple() in mushrooms or my_pos.y - 1 < 0
        
        if not up_blocked:
            # Check if up is safe from predicted death
            if not self.will_be_hit_soon(up_pos, horizon=2):
                if sandwiched or up_pos.to_tuple() not in self.danger_zones:
                    logger.debug(f"Bottom row escape: going UP (prediction-safe)")
                    return 'w'
            else:
                logger.debug(f"⚠️ Bottom row: UP blocked by predicted death at {up_pos.to_tuple()}")
        
        # Sideways escape - determine candidate directions
        if threats:
            closest_threat = min(threats, key=lambda x: abs(x - my_pos.x))
            
            # Prefer direction away from threat, but evaluate both
            if closest_threat < my_pos.x:
                # Threat on left, prefer right
                candidate_actions = ['d', 'a', 'w', '']
            else:
                # Threat on right, prefer left
                candidate_actions = ['a', 'd', 'w', '']
            
            # Use prediction-based filtering to choose safest action
            safest = self.get_safest_action_with_prediction(candidate_actions, my_pos)
            if safest:
                logger.debug(f"Bottom row escape: prediction-filtered choice '{safest}'")
                return safest
            
            # Fallback to find_safe_move which now includes prediction penalty
            if closest_threat < my_pos.x:
                return self.find_safe_move('d')
            else:
                return self.find_safe_move('a')
        
        return None
    
    def detect_horizontal_trap(self) -> Optional[str]:
        """
        Detect if we're trapped horizontally by mushrooms
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        
        # Validate bug_blaster has position
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        
        my_pos = Position(*bug_blaster['pos'])
        mushrooms = self.get_mushroom_positions()
        
        # Check if we're surrounded horizontally
        left_blocked = (my_pos.x - 1, my_pos.y) in mushrooms or my_pos.x - 1 < 0
        right_blocked = (my_pos.x + 1, my_pos.y) in mushrooms or my_pos.x + 1 >= self.map_size[0]
        
        if left_blocked and right_blocked:
            # Try to escape vertically
            up_pos = (my_pos.x, my_pos.y - 1)
            down_pos = (my_pos.x, my_pos.y + 1)
            
            if up_pos not in mushrooms and my_pos.y > 0:
                return 'w'
            elif down_pos not in mushrooms and my_pos.y < self.map_size[1] - 1:
                return 's'
            
            # Shoot to clear path
            if self.shot_cooldown == 0:
                return 'A'
        
        return None
    
    def is_safe_to_shoot(self) -> bool:
        """
        Verifica se é seguro atirar sem nos matar
        
        Regras:
        1. Se cobra está a 2+ linhas acima (y < my_y - 1) → sempre seguro
        2. Se cobra está exatamente 1 linha acima (y == my_y - 1):
           - Vai descer para a nossa linha após ser atingida
           - Vai inverter direção
           - Verificamos se podemos evitar a colisão
        3. Verifica se há caminho livre de cogumelos
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return False
        
        my_pos = Position(*bug_blaster['pos'])
        centipedes = self.game_state.get('centipedes', [])
        mushrooms = self.get_mushroom_positions()
        
        # Verificar se há alguma cobra na nossa coluna
        cobra_na_coluna = False
        
        for centipede in centipedes:
            body = centipede['body']
            direction = centipede['direction']  # 0=N, 1=E, 2=S, 3=W
            
            for idx, segment in enumerate(body):
                # Só importa se está na nossa coluna
                if segment[0] != my_pos.x:
                    continue
                
                cobra_na_coluna = True
                seg_y = segment[1]
                
                # Se está na mesma linha ou abaixo → não atirar
                if seg_y >= my_pos.y:
                    return False
                
                # Se está 2+ linhas acima → verificar caminho livre
                if seg_y < my_pos.y - 1:
                    # Verificar se há cogumelos no caminho
                    for y in range(seg_y + 1, my_pos.y):
                        if (my_pos.x, y) in mushrooms:
                            # Há cogumelo no caminho, mas podemos atirar para destruí-lo
                            # Só não é seguro se houver cobra muito perto
                            if seg_y > my_pos.y - 4:
                                return False
                    continue
                
                # Se está exatamente 1 linha acima → verificar segurança
                if seg_y == my_pos.y - 1:
                    # Após atirar, cobra desce para a nossa linha e inverte direção
                    # Precisamos garantir que podemos mover para o lado
                    
                    # Calcular direção invertida
                    if direction == 1:  # EAST -> WEST
                        cobra_next_x = my_pos.x - 1
                    elif direction == 3:  # WEST -> EAST
                        cobra_next_x = my_pos.x + 1
                    else:
                        # Se não está a mover horizontalmente, assumir perigo
                        return False
                    
                    # Verificar se podemos mover para o lado oposto
                    if direction == 1:  # Cobra vai para WEST, nós movemos EAST
                        safe_pos = (my_pos.x + 1, my_pos.y)
                    else:  # Cobra vai para EAST, nós movemos WEST
                        safe_pos = (my_pos.x - 1, my_pos.y)
                    
                    # Verificar se a posição segura é válida
                    if (safe_pos[0] < 0 or safe_pos[0] >= self.map_size[0] or
                        safe_pos in mushrooms or
                        safe_pos in self.danger_zones):
                        return False
                    
                    # CRITICAL: Check if escape position will be hit by predicted centipede movement
                    safe_position = Position(safe_pos[0], safe_pos[1])
                    if self.will_be_hit_soon(safe_position, horizon=2):
                        logger.debug(f"⚠️ Shoot unsafe: escape position {safe_pos} will be hit soon")
                        return False
        
        # Se há cobra na coluna, já verificamos a segurança acima
        # Se não há cobra, podemos atirar livremente (para limpar cogumelos)
        return True
    
    def emergency_evade(self) -> Optional[str]:
        """
        Emergency evasion when in immediate danger
        
        CRITICAL IMPROVEMENT: Uses prediction-based safety to avoid fleeing into
        a position where a centipede head will arrive in the next frame.
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        
        # Validate bug_blaster has position
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        
        my_pos = Position(*bug_blaster['pos'])
        
        centipedes = self.game_state.get('centipedes', [])
        
        # Check if any centipede segment is adjacent
        immediate_threat = False
        for centipede in centipedes:
            for segment in centipede['body']:
                seg_pos = Position(*segment)
                if my_pos.manhattan_distance(seg_pos) <= 1:
                    immediate_threat = True
                    break
            if immediate_threat:
                break
        
        if not immediate_threat:
            return None
        
        logger.warning(f"EMERGENCY! Centipede adjacent to position {my_pos.to_tuple()}")
        
        # Use prediction-based filtering to find safest escape
        # Try all directions plus staying still
        candidate_actions = ['w', 'a', 's', 'd', '']
        
        safest = self.get_safest_action_with_prediction(candidate_actions, my_pos)
        if safest:
            logger.info(f"Emergency evade: prediction-filtered choice '{safest}'")
            return safest
        
        # Fallback to find_safe_move which includes prediction penalty
        return self.find_safe_move()
    
    def check_camping_threats(self) -> Optional[str]:
        """
        Check for threats that should break camping mode.
        
        Threats considered:
        1. Free centipedes approaching in same or adjacent column (within 5 rows)
        2. Free centipedes within 3 rows and descending
        3. Spiders in danger zone (not implemented - no spider data yet)
        4. Fleas falling directly on agent's column (not implemented - no flea data yet)
        5. Stuck centipede target becomes unstuck
        
        Returns:
            - None if no threats (safe to continue camping)
            - 'unstuck' if target became unstuck
            - 'threat' if there's a threat requiring evasion
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return 'threat'
        
        my_pos = Position(*bug_blaster['pos'])
        centipedes = self.game_state.get('centipedes', [])
        
        # Check if camping target still exists and is still stuck
        if self.camping_target_name:
            target_exists = False
            target_still_stuck = False
            
            for centipede in centipedes:
                if centipede['name'] == self.camping_target_name:
                    target_exists = True
                    if self.camping_target_name in self.stuck_centipedes:
                        target_still_stuck = True
                    break
            
            # Target died - will need new target, but not a threat per se
            if not target_exists:
                logger.info(f"🏕️ Camping target '{self.camping_target_name}' died")
                return 'unstuck'
            
            # Target became unstuck - need to exit camping
            if not target_still_stuck:
                logger.info(f"🏕️ Camping target '{self.camping_target_name}' is no longer stuck")
                return 'unstuck'
        
        # Check for free centipede threats
        for centipede in centipedes:
            name = centipede['name']
            body = centipede['body']
            
            # Skip stuck centipedes - they're not threats
            if name in self.stuck_centipedes:
                continue
            
            if not body:
                continue
            
            # Get head position (last element)
            head = Position(*body[-1])
            
            # Threat 1: Free centipede in same or adjacent column and within 5 rows
            column_distance = abs(head.x - my_pos.x)
            row_distance = my_pos.y - head.y  # Positive if centipede is above us
            
            if column_distance <= 1 and 0 < row_distance <= 5:
                logger.warning(f"🏕️ CAMPING THREAT: Free centipede '{name}' at column distance {column_distance}, {row_distance} rows above")
                return 'threat'
            
            # Threat 2: Free centipede within 3 rows (any column) and potentially descending
            if 0 < row_distance <= 3:
                # Check if it's moving toward us (descending or same row)
                direction = centipede.get('direction', 1)  # 0=N, 1=E, 2=S, 3=W
                
                # If centipede is close horizontally, it's a threat
                if column_distance <= 3:
                    logger.warning(f"🏕️ CAMPING THREAT: Free centipede '{name}' very close - {row_distance} rows, {column_distance} cols")
                    return 'threat'
            
            # Threat 3: Any free centipede on same row or below us
            if row_distance <= 0 and column_distance <= 4:
                logger.warning(f"🏕️ CAMPING THREAT: Free centipede '{name}' at same level or below, column distance {column_distance}")
                return 'threat'
        
        # No threats detected
        return None
    
    def should_enter_camping(self, target_name: str, is_aligned: bool, is_stuck: bool) -> bool:
        """
        Determine if agent should enter camping mode.
        
        Mid game: Only camp if aligned, no threats, clearly safe
        Late game: Auto-enter when aligned with stuck target
        
        Args:
            target_name: Name of the target centipede
            is_aligned: True if agent is horizontally aligned with target
            is_stuck: True if target is a stuck centipede
        
        Returns:
            True if should enter camping mode
        """
        if not is_stuck or not is_aligned:
            return False
        
        # Check for any threats
        threat = self.check_camping_threats()
        if threat:
            return False
        
        # Late game: Always camp against stuck targets when aligned and safe
        if self.late_game:
            return True
        
        # Mid game: More conservative - only camp if clearly safe
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return False
        
        my_pos = Position(*bug_blaster['pos'])
        centipedes = self.game_state.get('centipedes', [])
        
        # Check if there are any free centipedes nearby
        for centipede in centipedes:
            if centipede['name'] in self.stuck_centipedes:
                continue
            
            if not centipede['body']:
                continue
            
            head = Position(*centipede['body'][-1])
            distance = my_pos.manhattan_distance(head)
            
            # In mid game, don't camp if any free centipede within 8 units
            if distance < 8:
                logger.debug(f"Mid game: not camping - free centipede '{centipede['name']}' at distance {distance}")
                return False
        
        # Safe to camp in mid game
        return True
    
    def decide_camping_action(self) -> Optional[str]:
        """
        Camping mode behavior for targeting stuck centipedes.
        
        Behavior:
        - Stay stationary
        - Shoot continuously when cooldown allows and line is clear
        - Maintain defensive awareness for threats
        - Exit camping if threats detected or target unstuck
        
        Returns:
            - Action string if camping should continue
            - None if camping should exit
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            self.camping_mode = False
            return None
        
        my_pos = Position(*bug_blaster['pos'])
        
        # Check for threats that should break camping
        threat = self.check_camping_threats()
        if threat:
            if threat == 'threat':
                logger.info(f"🏕️ EXITING CAMPING - threat detected, evading")
                self.camping_mode = False
                self.camping_target_name = None
                # Return evasive action
                return self.find_safe_move()
            elif threat == 'unstuck':
                logger.info(f"🏕️ EXITING CAMPING - target no longer stuck")
                self.camping_mode = False
                self.camping_target_name = None
                return None  # Let normal logic take over
        
        # Validate camping target still exists
        centipedes = self.game_state.get('centipedes', [])
        target_centipede = None
        for centipede in centipedes:
            if centipede['name'] == self.camping_target_name:
                target_centipede = centipede
                break
        
        if not target_centipede or not target_centipede['body']:
            logger.info(f"🏕️ EXITING CAMPING - target '{self.camping_target_name}' no longer exists")
            self.camping_mode = False
            self.camping_target_name = None
            return None
        
        target_head = Position(*target_centipede['body'][-1])
        mushrooms = self.get_mushroom_positions()
        
        # Check if still aligned
        if my_pos.x != target_head.x:
            logger.info(f"🏕️ EXITING CAMPING - no longer aligned (my_x={my_pos.x}, target_x={target_head.x})")
            self.camping_mode = False
            self.camping_target_name = None
            return None
        
        # Check line of fire
        mushrooms_in_path = 0
        for y in range(target_head.y + 1, my_pos.y):
            if (my_pos.x, y) in mushrooms:
                mushrooms_in_path += 1
        
        line_clear = (mushrooms_in_path == 0)
        
        # Shoot if possible
        if self.shot_cooldown == 0:
            if line_clear:
                # Direct shot at target
                if self.is_safe_to_shoot():
                    logger.info(f"🏕️ CAMPING: shooting stuck target '{self.camping_target_name}'")
                    self.shot_cooldown = 10
                    return 'A'
            elif mushrooms_in_path <= 3:
                # Clear mushrooms blocking path
                if self.is_safe_to_shoot():
                    logger.info(f"🏕️ CAMPING: clearing {mushrooms_in_path} mushroom(s) to reach '{self.camping_target_name}'")
                    self.shot_cooldown = 10
                    return 'A'
            else:
                # Too many mushrooms - need to reposition
                logger.info(f"🏕️ EXITING CAMPING - {mushrooms_in_path} mushrooms blocking, need reposition")
                self.camping_mode = False
                self.camping_target_name = None
                return None
        
        # Stay still while waiting for cooldown
        # Return empty string to indicate "do nothing" (stay in place)
        self.debug_info['reason'] = 'camping_waiting'
        return ''

    def decide_focus_hunt_action(self) -> Optional[str]:
        """
        Late game focused hunting behavior
        - Follow focus target closely
        - Stay in safe zone (bottom 5-6 rows)
        - Align horizontally with target
        - Shoot only when line is clear
        - Never shoot to clear mushrooms in late game
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        
        my_pos = Position(*bug_blaster['pos'])
        
        # Update focus target if needed
        if not self.focus_target_name:
            self.update_focus_target()
        
        # Validate focus target still exists
        centipedes = self.game_state.get('centipedes', [])
        focus_centipede = None
        for centipede in centipedes:
            if centipede['name'] == self.focus_target_name:
                focus_centipede = centipede
                break
        
        # If focus target died, choose new one
        if not focus_centipede:
            logger.debug(f"Focus target '{self.focus_target_name}' died, selecting new target")
            self.focus_target_name = None
            self.update_focus_target()
            
            # Try again with new target
            for centipede in centipedes:
                if centipede['name'] == self.focus_target_name:
                    focus_centipede = centipede
                    break
        
        # If still no target, return None
        if not focus_centipede:
            logger.debug("No focus target available in late game")
            return None
        
        focus_body = focus_centipede['body']
        if not focus_body:
            return None
        
        focus_head = Position(*focus_body[-1])  # Head is last element
        mushrooms = self.get_mushroom_positions()
        
        # Check if corridor is too tight (too many mushrooms around)
        tight_corridor = False
        mushrooms_nearby = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_pos = (my_pos.x + dx, my_pos.y + dy)
                if check_pos in mushrooms:
                    mushrooms_nearby += 1
        
        if mushrooms_nearby >= 5:
            tight_corridor = True
            logger.warning("Late game: corridor too tight, temporarily abandoning focus target")
            self.focus_target_name = None
            return self.find_safe_move()
        
        # Stay in safe zone (bottom 5-6 rows)
        safe_zone_limit = self.map_size[1] - 6
        if my_pos.y < safe_zone_limit:
            # Move down to safe zone
            down_pos = Position(my_pos.x, my_pos.y + 1)
            if self.game_state.get('bug_blaster') and (down_pos.x, down_pos.y) not in mushrooms:
                if (down_pos.x, down_pos.y) not in self.danger_zones:
                    logger.debug(f"Late game: moving to safe zone from y={my_pos.y}")
                    return 's'
        
        # Check if aligned with focus target
        if my_pos.x == focus_head.x and focus_head.y < my_pos.y:
            is_stuck_target = self.focus_target_name in self.stuck_centipedes
            
            # CAMPING MODE ENTRY POINT
            # If aligned with stuck target, consider entering camping mode
            if is_stuck_target and self.should_enter_camping(self.focus_target_name, True, True):
                if not self.camping_mode:
                    logger.info(f"🏕️ ENTERING CAMPING MODE against stuck '{self.focus_target_name}' at column {focus_head.x}")
                    self.camping_mode = True
                    self.camping_target_name = self.focus_target_name
                    self.camping_start_frame = self.frame_count
                
                # Let camping mode handle the action
                action = self.decide_camping_action()
                if action is not None:
                    return action
                # If camping exited, continue with normal logic below
            
            # Check if line is clear and count mushrooms
            mushrooms_in_path = 0
            for y in range(focus_head.y + 1, my_pos.y):
                if (my_pos.x, y) in mushrooms:
                    mushrooms_in_path += 1
            
            line_clear = (mushrooms_in_path == 0)
            
            if line_clear and self.shot_cooldown == 0:
                # Safe to shoot?
                if self.is_safe_to_shoot():
                    logger.info(f"🎯 Late game: shooting focus target '{self.focus_target_name}' at {focus_head.to_tuple()}")
                    self.shot_cooldown = 10
                    return 'A'
            
            # LATE GAME CLEARING LOGIC FOR STUCK CENTIPEDES
            # If target is a stuck centipede and mushrooms block the path,
            # allow clearing shots to open the line (max 3 mushrooms)
            if not line_clear:
                if is_stuck_target and mushrooms_in_path <= 3 and self.shot_cooldown == 0:
                    # Targeting stuck centipede with blocked path - clear mushrooms
                    if self.is_safe_to_shoot():
                        logger.info(f"🎯 Late game focus hunt: clearing {mushrooms_in_path} mushroom(s) to reach stuck '{self.focus_target_name}'")
                        self.shot_cooldown = 10
                        return 'A'
                elif is_stuck_target and mushrooms_in_path > 3:
                    # Too many mushrooms to clear - seek different column
                    logger.debug(f"Late game: {mushrooms_in_path} mushrooms blocking stuck target, seeking new position")
                    if my_pos.x < self.map_size[0] // 2:
                        return self.find_safe_move('d')
                    else:
                        return self.find_safe_move('a')
                else:
                    # Not targeting stuck centipede - move to find clear column
                    logger.debug(f"Late game: line blocked by mushrooms, seeking new position")
                    if my_pos.x < self.map_size[0] // 2:
                        return self.find_safe_move('d')
                    else:
                        return self.find_safe_move('a')
        
        # Not aligned - always align horizontally with focus target column
        # regardless of how high the target is. Trust find_safe_move and danger zones
        # for safety. The goal is to be ready to shoot when opportunity arises.
        
        is_stuck_target = self.focus_target_name in self.stuck_centipedes
        
        if my_pos.x < focus_head.x:
            logger.debug(f"Late game: aligning right towards focus target column {focus_head.x} (stuck={is_stuck_target})")
            return self.find_safe_move('d')
        elif my_pos.x > focus_head.x:
            logger.debug(f"Late game: aligning left towards focus target column {focus_head.x} (stuck={is_stuck_target})")
            return self.find_safe_move('a')
        
        # Already aligned but can't shoot (maybe waiting for cooldown or not safe)
        # Stay in position - find_safe_move with no preference will keep us safe
        return self.find_safe_move()
    
    def decide_action(self) -> str:
        """
        Main decision logic - priority-based
        Order:
        1. Bottom-row escape
        2. Horizontal trap escape
        3. Emergency evade
        4. Defensive strategy movement
        5. Aggressive shooting (align, shoot, move)
        6. Clearing mushrooms
        7. Return to home row
        8. Center horizontally  (AGORA DESATIVADO NO CÓDIGO)
        9. Fallback safe move
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        
        # Validate bug_blaster has position
        if not bug_blaster or 'pos' not in bug_blaster:
            return ''
        
        my_pos = Position(*bug_blaster['pos'])
        
        # 1. Bottom-row escape
        if action := self.evade_bottom_row_snake():
            self.debug_info['reason'] = 'bottom_row_escape'
            logger.info(f"Bottom-row escape: {action}")
            return action
        
        # 2. Horizontal trap escape
        if action := self.detect_horizontal_trap():
            self.debug_info['reason'] = 'horizontal_trap'
            logger.info(f"Horizontal trap escape: {action}")
            return action
        
        # 3. Emergency evade
        if action := self.emergency_evade():
            self.debug_info['reason'] = 'emergency_evade'
            logger.info(f"Emergency evade: {action}")
            # Exit camping mode on emergency
            if self.camping_mode:
                logger.info("🏕️ EXITING CAMPING - emergency evade triggered")
                self.camping_mode = False
                self.camping_target_name = None
            return action
        
        # 3.3 CAMPING MODE - Stay stationary and shoot stuck centipedes
        # This has high priority to prevent unnecessary movement when camping
        if self.camping_mode:
            action = self.decide_camping_action()
            if action is not None:
                self.debug_info['reason'] = 'camping_mode'
                return action
            # If camping returned None, it exited - continue to normal logic
        
        # 3.5 ADADADA LOOP BREAKER - High priority when oscillating with stuck centipede target
        # If we're stuck in horizontal oscillation AND targeting a stuck centipede,
        # try to clear the path by shooting instead of continuing to oscillate
        if self.self_stuck_detected and self.late_game:
            centipedes = self.game_state.get('centipedes', [])
            target = self.find_best_target()
            
            if target:
                target_name, target_pos, _ = target
                target_x, target_y = target_pos
                
                # Check if targeting a stuck centipede
                if target_name in self.stuck_centipedes:
                    mushrooms = self.get_mushroom_positions()
                    
                    # Count mushrooms in path from current column
                    mushrooms_in_path = 0
                    for y in range(target_y + 1, my_pos.y):
                        if (my_pos.x, y) in mushrooms:
                            mushrooms_in_path += 1
                    
                    # If we're nearly aligned (within 2 columns) and there are mushrooms blocking
                    distance_x = abs(my_pos.x - target_x)
                    if distance_x <= 2 and mushrooms_in_path > 0 and mushrooms_in_path <= 3:
                        if self.shot_cooldown == 0 and self.is_safe_to_shoot():
                            self.debug_info['reason'] = 'adadada_loop_breaker'
                            self.shot_cooldown = 10
                            logger.info(f"🎯 ADADADA LOOP BREAKER: shooting to clear {mushrooms_in_path} mushroom(s) blocking stuck target '{target_name}' (distance_x={distance_x})")
                            return 'A'
                    # If we're aligned but path is clear, just shoot
                    elif my_pos.x == target_x and mushrooms_in_path == 0:
                        if self.shot_cooldown == 0 and self.is_safe_to_shoot():
                            self.debug_info['reason'] = 'adadada_loop_breaker_direct'
                            self.shot_cooldown = 10
                            logger.info(f"🎯 ADADADA LOOP BREAKER: direct shot at stuck target '{target_name}'")
                            return 'A'
        
        # 4. Defensive strategy
        if self.current_strategy == "defensive":
            self.debug_info['reason'] = 'defensive_mode'
            
            # Tentar atirar se for seguro E estamos alinhados com um alvo
            target = self.find_best_target()
            if target and self.shot_cooldown == 0:
                target_name, target_pos, target_score = target
                target_x, target_y = target_pos
                
                # Estamos alinhados?
                if my_pos.x == target_x:
                    # É seguro atirar?
                    if self.is_safe_to_shoot():
                        self.debug_info['reason'] = 'defensive_safe_shot'
                        self.shot_cooldown = 10
                        logger.debug(f"Defensive safe shot at {target_pos}")
                        return 'A'
                else:
                    # Não alinhados - tentar alinhar se for seguro
                    # Só alinhar se o alvo está a 3+ linhas acima (seguro)
                    if target_y < my_pos.y - 2:
                        if my_pos.x < target_x:
                            self.debug_info['reason'] = 'defensive_align_right'
                            move = self.find_safe_move('d')
                            if move == 'd':  # Só move se for seguro
                                logger.debug(f"Defensive align right towards {target_pos}")
                                return move
                        elif my_pos.x > target_x:
                            self.debug_info['reason'] = 'defensive_align_left'
                            move = self.find_safe_move('a')
                            if move == 'a':  # Só move se for seguro
                                logger.debug(f"Defensive align left towards {target_pos}")
                                return move
            
            # Mover para safe zone e evadir
            if my_pos.y < self.safe_zone_start:
                action = self.find_safe_move('s')
                logger.debug(f"Defensive: moving to safe zone")
            else:
                action = self.find_safe_move()
                logger.debug(f"Defensive: safe movement")
            return action
        
        # 5. Aggressive shooting / Late game focus hunting
        if self.current_strategy == "aggressive":
            # In late game, use focused hunting behavior
            if self.late_game:
                self.debug_info['reason'] = 'late_game_focus_hunt'
                action = self.decide_focus_hunt_action()
                if action:
                    return action
                # If focus hunt returns None, fall through to normal aggressive
            
            # Normal aggressive behavior (or late game fallback)
            target = self.find_best_target()
            
            if target:
                target_name, target_pos, target_score = target
                target_x, target_y = target_pos
                
                # Calculate horizontal distance to target
                distance_x = abs(my_pos.x - target_x)
                
                # MID-GAME CAMPING MODE CHECK
                # If aligned with stuck centipede and safe, enter camping
                is_stuck_target = target_name in self.stuck_centipedes
                is_aligned = (distance_x == 0)
                
                if is_aligned and is_stuck_target and not self.late_game:
                    if self.should_enter_camping(target_name, True, True):
                        if not self.camping_mode:
                            logger.info(f"🏕️ MID-GAME: ENTERING CAMPING MODE against stuck '{target_name}'")
                            self.camping_mode = True
                            self.camping_target_name = target_name
                            self.camping_start_frame = self.frame_count
                        
                        action = self.decide_camping_action()
                        if action is not None:
                            self.debug_info['reason'] = 'midgame_camping'
                            return action
                        # If camping exited, continue with normal logic
                
                # Improved shooting vs aligning decision
                # ==========================================                
                mushrooms = self.get_mushroom_positions()
                
                # Check if path would be clear from current position
                current_path_clear = True
                if my_pos.x == target_x:
                    for y in range(target_y + 1, my_pos.y):
                        if (my_pos.x, y) in mushrooms:
                            current_path_clear = False
                            break
                
                # Define "reasonably aligned" threshold
                # - Perfectly aligned (distance_x == 0): always try to shoot if safe
                # - Self-stuck detected: be more permissive (allow distance_x <= 1)
                # - Normal: only shoot when perfectly aligned (distance_x == 0)
                reasonably_aligned_threshold = 1 if self.self_stuck_detected else 0
                
                is_reasonably_aligned = distance_x <= reasonably_aligned_threshold
                
                # Count mushrooms in path
                mushrooms_in_path = 0
                for y in range(target_y + 1, my_pos.y):
                    if (my_pos.x, y) in mushrooms:
                        mushrooms_in_path += 1
                
                # Decision: Should we shoot or continue aligning?
                if is_reasonably_aligned and self.shot_cooldown == 0:
                    # Check path from current column
                    path_clear_from_here = mushrooms_in_path == 0
                    
                    if path_clear_from_here:
                        # Shoot from current position
                        self.debug_info['reason'] = 'shooting_target'
                        self.shot_cooldown = 10
                        if self.self_stuck_detected and distance_x > 0:
                            logger.info(f"🎯 Shooting from reasonably aligned position (distance_x={distance_x}) - self-stuck mitigation")
                        logger.debug(f"Aggressive shot at {target_pos}, score={target_score:.1f}")
                        return 'A'
                    elif self.self_stuck_detected and mushrooms_in_path == 1 and target_name in self.stuck_centipedes:
                        # Special case: self-stuck + targeting stuck centipede + only 1 mushroom blocking
                        # Favor shooting to clear over endless horizontal adjustments
                        centipedes = self.game_state.get('centipedes', [])
                        all_stuck = all(c['name'] in self.stuck_centipedes for c in centipedes)
                        if all_stuck and self.is_safe_to_shoot():
                            self.debug_info['reason'] = 'self_stuck_single_mushroom_clear'
                            self.shot_cooldown = 10
                            logger.info(f"🎯 Self-stuck mitigation: shooting through single blocking mushroom (distance_x={distance_x})")
                            return 'A'
                    elif not self.late_game and my_pos.x == target_x:
                        # Only shoot to clear path when perfectly aligned and not in late game
                        self.debug_info['reason'] = 'clearing_shot_path'
                        self.shot_cooldown = 10
                        logger.debug(f"Clearing mushroom in shot path to {target_pos}")
                        return 'A'
                    elif self.late_game and my_pos.x == target_x:
                        # LATE GAME CLEARING SHOTS FOR STUCK CENTIPEDES
                        centipedes = self.game_state.get('centipedes', [])
                        all_stuck = all(c['name'] in self.stuck_centipedes for c in centipedes)
                        if target_name in self.stuck_centipedes:
                            if mushrooms_in_path <= 3 and self.is_safe_to_shoot():
                                self.debug_info['reason'] = 'late_game_clearing_for_stuck_target'
                                self.shot_cooldown = 10
                                logger.info(f"🎯 Late game: clearing {mushrooms_in_path} mushroom(s) to reach stuck centipede '{target_name}'")
                                return 'A'
                            elif mushrooms_in_path > 3:
                                logger.debug(f"Late game: too many mushrooms ({mushrooms_in_path}) blocking stuck target, seeking new position")
                        elif all_stuck and self.self_stuck_detected:
                            if self.is_safe_to_shoot():
                                self.debug_info['reason'] = 'late_game_unstuck_clearing_shot'
                                self.shot_cooldown = 10
                                logger.info(f"🎯 Late game exception: clearing shot to unstuck (all stuck + self-stuck)")
                                return 'A'
                
                # Not reasonably aligned yet - continue aligning
                if my_pos.x < target_x:
                    self.debug_info['reason'] = 'aligning_right'
                    logger.debug(f"Aligning right to target at x={target_x} (current distance: {distance_x})")
                    return self.find_safe_move('d')
                elif my_pos.x > target_x:
                    self.debug_info['reason'] = 'aligning_left'
                    logger.debug(f"Aligning left to target at x={target_x} (current distance: {distance_x})")
                    return self.find_safe_move('a')
        
        # 6. Clearing mushrooms
        if self.current_strategy == "clearing":
            self.debug_info['reason'] = 'clearing_mode'
            logger.debug("Clearing mode: shooting mushrooms")
            # Shoot upward to clear mushrooms
            if self.shot_cooldown == 0:
                self.shot_cooldown = 10
                return 'A'
            # Move around while waiting
            return self.find_safe_move()
        
        # 7. Return to home row (safe zone)
        if my_pos.y < self.safe_zone_start:
            self.debug_info['reason'] = 'return_to_safe_zone'
            logger.info(f"🏠 Returning to safe zone from y={my_pos.y} to y>={self.safe_zone_start}")
            # CRITICAL: Pass returning_to_safe_zone=True to override normal distance-based safety
            return self.find_safe_move('s', returning_to_safe_zone=True)
        
        # 8. Center horizontally (DESATIVADO)
        # Toda a lógica de recentrar no meio do mapa foi comentada para não
        # afastar o agente de cobras presas / posições vantajosas.
        #
        # center_x = self.map_size[0] // 2
        # if abs(my_pos.x - center_x) > 3:
        #     if my_pos.x < center_x:
        #         self.debug_info['reason'] = 'centering_right'
        #         logger.debug(f"Centering: moving right from x={my_pos.x}")
        #         return self.find_safe_move('d')
        #     else:
        #         self.debug_info['reason'] = 'centering_left'
        #         logger.debug(f"Centering: moving left from x={my_pos.x}")
        #         return self.find_safe_move('a')
        
        # 9. Fallback
        self.debug_info['reason'] = 'fallback_safe'
        logger.debug("Fallback: safe movement")
        return self.find_safe_move()
    
    def update_focus_target(self):
        """
        Update focus target for late game hunting
        Priority:
        1. Stuck centipedes - scored by: length, Y position (lower is better), column distance
        2. Longest centipede overall
        3. Tie-break: closest to player, not too high, accessible column
        
        In late game, stuck centipedes are ALWAYS prioritized as they are easier targets.
        Among stuck centipedes, prefer those that are:
        - Lower in the map (higher Y coordinate) - easier to reach and kill
        - Longer (more potential points)
        - Closer to our current column (faster alignment)
        """
        centipedes = self.game_state.get('centipedes', [])
        if not centipedes:
            self.focus_target_name = None
            return
        
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return
        
        my_pos = Position(*bug_blaster['pos'])
        
        # Separate stuck and non-stuck centipedes
        stuck_centipedes = []
        normal_centipedes = []
        
        for centipede in centipedes:
            name = centipede['name']
            body = centipede['body']
            length = len(body)
            
            if name in self.stuck_centipedes:
                stuck_centipedes.append((name, body, length))
            else:
                normal_centipedes.append((name, body, length))
        
        # Priority 1: Choose best stuck centipede
        # In late game, stuck centipedes are the primary target
        # Score by: length, Y position (lower is better), column distance (closer is better)
        if stuck_centipedes:
            best_stuck = None
            best_stuck_score = -float('inf')
            
            for name, body, length in stuck_centipedes:
                if not body:
                    continue
                
                head = Position(*body[-1])  # Head is last element
                
                # Length score: more segments = more points potential
                length_score = length * 10
                
                # Y position score: lower in map (higher Y) = easier to kill
                # This is critical - stuck centipedes high up are hard to reach
                y_score = head.y * 5
                
                # Column distance: closer to our column = faster alignment
                column_dist = abs(head.x - my_pos.x)
                column_score = -column_dist * 3
                
                total_score = length_score + y_score + column_score
                
                if total_score > best_stuck_score:
                    best_stuck_score = total_score
                    best_stuck = (name, body, length)
            
            if best_stuck:
                old_target = self.focus_target_name
                self.focus_target_name = best_stuck[0]
                if old_target != best_stuck[0]:
                    head_y = best_stuck[1][-1][1] if best_stuck[1] else 0
                    logger.info(f"🎯 Late game focus: STUCK centipede '{best_stuck[0]}' (length: {best_stuck[2]}, y: {head_y}, score: {best_stuck_score:.1f})")
                return
        
        # Priority 2: Choose longest centipede overall
        all_centipedes = normal_centipedes
        if not all_centipedes:
            self.focus_target_name = None
            return
        
        # Sort by length (descending)
        all_centipedes.sort(key=lambda x: x[2], reverse=True)
        
        # Get all with max length for tie-breaking
        max_length = all_centipedes[0][2]
        candidates = [c for c in all_centipedes if c[2] == max_length]
        
        # Tie-break by distance, height, and accessibility
        best_candidate = None
        best_score = -float('inf')
        
        for name, body, length in candidates:
            if not body:
                continue
            
            head = Position(*body[-1])  # Head is last element
            
            # Distance score (closer is better)
            distance = my_pos.distance(head)
            distance_score = -distance
            
            # Height score (not too high, prefer middle-low area)
            # Penalty for being too high (y < 10)
            if head.y < 10:
                height_score = -50
            else:
                height_score = 0
            
            # Accessibility score (how close to our column)
            column_distance = abs(head.x - my_pos.x)
            accessibility_score = -column_distance * 2
            
            total_score = distance_score + height_score + accessibility_score
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = name
        
        old_target = self.focus_target_name
        self.focus_target_name = best_candidate
        if old_target != best_candidate:
            logger.info(f"🎯 Late game focus: centipede '{best_candidate}' (length: {max_length})")
    
    def get_mushroom_positions(self) -> Set[Tuple[int, int]]:
        """Helper to get all mushroom positions as set of tuples"""
        mushrooms = self.game_state.get('mushrooms', [])
        return {tuple(m['pos']) for m in mushrooms}


async def agent_loop(server_address="localhost:8000", agent_name="student"):
    """Main agent loop"""
    import websockets
    
    agent = CentipedeAgent()
    
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        # Join game
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))
        logger.info(f"Agent {agent_name} joined the game")
        
        while True:
            try:
                # Receive game state
                state = json.loads(await websocket.recv())
                
                # Update agent state
                agent.update_state(state)
                
                # Log important events periodically
                if agent.frame_count % 50 == 0:
                    score = state.get('score', 0)
                    num_centipedes = len(state.get('centipedes', []))
                    num_mushrooms = len(state.get('mushrooms', []))
                    logger.info(f"Step: {agent.frame_count}, Score: {score}, "
                              f"Strategy: {agent.current_strategy}, "
                              f"Centipedes: {num_centipedes}, "
                              f"Mushrooms: {num_mushrooms}")
                
                # Decide action with error handling
                try:
                    key = agent.decide_action()
                    agent.last_action = key
                except Exception as action_error:
                    # Log the error but don't crash
                    logger.error(f"Error deciding action: {action_error}", exc_info=True)
                    # Fallback: safe movement
                    key = agent.find_safe_move()
                    if not key:
                        key = ''
                
                # Send action
                await websocket.send(json.dumps({"cmd": "key", "key": key}))
                
                # Debug output with reason
                if agent.frame_count % 100 == 0 and agent.debug_info.get('reason'):
                    logger.debug(f"Action: {key}, Reason: {agent.debug_info.get('reason')}, "
                               f"Position: {agent.game_state.get('bug_blaster', {}).get('pos', 'unknown')}")
                
            except websockets.exceptions.ConnectionClosedOK:
                logger.info("Server has cleanly disconnected us")
                return
            except Exception as e:
                logger.error(f"Error in agent loop: {e}", exc_info=True)
                break


# Entry point
if __name__ == "__main__":
    SERVER = os.environ.get("SERVER", "localhost")
    PORT = os.environ.get("PORT", "8000")
    NAME = os.environ.get("NAME", getpass.getuser())
    
    try:
        asyncio.run(agent_loop(f"{SERVER}:{PORT}", NAME))
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)