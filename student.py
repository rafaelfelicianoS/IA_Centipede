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
"""

import asyncio
import getpass
import json
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
import math

# Game constants
DIRECTIONS = {
    'w': (0, -1),  # UP
    's': (0, 1),   # DOWN
    'a': (-1, 0),  # LEFT
    'd': (1, 0),   # RIGHT
}

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
            self.current_strategy = "defensive"
            return
        
        # CLEARING: Too many mushrooms (> 150)
        if mushroom_count > 150:
            self.current_strategy = "clearing"
            return
        
        # AGGRESSIVE: Default - hunt and kill
        self.current_strategy = "aggressive"
    
    def find_best_target(self) -> Optional[Tuple[str, Tuple[int, int], float]]:
        """
        Find the best centipede segment to target
        Returns: (centipede_name, segment_position, score) or None
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        
        # Validate bug_blaster has position
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        
        my_pos = Position(*bug_blaster['pos'])
        
        centipedes = self.game_state.get('centipedes', [])
        mushrooms = self.get_mushroom_positions()
        
        targets = []
        
        for centipede in centipedes:
            name = centipede['name']
            body = centipede['body']
            
            for idx, segment in enumerate(body):
                seg_pos = Position(*segment)
                
                # Only target segments above us
                if seg_pos.y >= my_pos.y:
                    continue
                
                score = 0
                
                # Height score (higher = more points in game)
                score += (self.map_size[1] - seg_pos.y) * 10
                
                # Head bonus (last element in body)
                if idx == len(body) - 1:
                    score += 50
                
                # Column alignment bonus
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
                
                # Proximity penalty (too close is dangerous)
                dist = my_pos.manhattan_distance(seg_pos)
                if dist < 3:
                    score -= 100
                
                # Prediction bonus (target predicted positions)
                predictions = self.predicted_positions.get(name, [])
                if tuple(segment) in predictions[:2]:
                    score += 30
                
                # Centipede stuck bonus (easier to hit)
                if len(self.centipede_tracking[name]) >= 3:
                    recent_positions = list(self.centipede_tracking[name])[-3:]
                    if len(set(map(tuple, recent_positions))) == 1:  # Not moving
                        score += 40
                
                targets.append((name, tuple(segment), score))
        
        if not targets:
            return None
        
        # Sort by score (highest first)
        targets.sort(key=lambda x: x[2], reverse=True)
        return targets[0]
    
    def find_safe_move(self, preferred_direction: Optional[str] = None) -> str:
        """
        Find the safest move by scoring all possible actions
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        
        # Validate bug_blaster has position
        if not bug_blaster or 'pos' not in bug_blaster:
            return ''
        
        my_pos = Position(*bug_blaster['pos'])
        
        mushrooms = self.get_mushroom_positions()
        centipedes = self.game_state.get('centipedes', [])
        
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
                score -= 500
                continue
            
            # Danger zone?
            if new_pos.to_tuple() in self.danger_zones:
                score -= 300
            
            # Distance to nearest centipede (bigger = better)
            min_centipede_dist = float('inf')
            for centipede in centipedes:
                for segment in centipede['body']:
                    seg_pos = Position(*segment)
                    dist = new_pos.manhattan_distance(seg_pos)
                    min_centipede_dist = min(min_centipede_dist, dist)
            
            score += min_centipede_dist * 20
            
            # Stay in safe zone bonus
            if new_pos.y >= self.safe_zone_start:
                score += 50
            
            # Move toward center bonus (avoid edges)
            center_x = self.map_size[0] // 2
            distance_to_center = abs(new_pos.x - center_x)
            score -= distance_to_center * 2
            
            # Prefer requested direction
            if preferred_direction and action == preferred_direction:
                score += 100
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def evade_bottom_row_snake(self) -> Optional[str]:
        """
        Special case: Avoid dying when cornered on last row
        Only active on last row
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
        
        # Try to go UP
        up_pos = (my_pos.x, my_pos.y - 1)
        if up_pos not in mushrooms:
            if sandwiched:
                # Go up even if danger zone (emergency)
                return 'w'
            elif up_pos not in self.danger_zones:
                return 'w'
        
        # Move sideways away from closest segment
        if threats:
            closest_threat = min(threats, key=lambda x: abs(x - my_pos.x))
            if closest_threat < my_pos.x:
                # Threat on left, go right
                return self.find_safe_move('d')
            else:
                # Threat on right, go left
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
        
        # Se há cobra na coluna, já verificamos a segurança acima
        # Se não há cobra, podemos atirar livremente (para limpar cogumelos)
        return True
    
    def emergency_evade(self) -> Optional[str]:
        """
        Emergency evasion when in immediate danger
        """
        bug_blaster = self.game_state.get('bug_blaster', {})
        
        # Validate bug_blaster has position
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        
        my_pos = Position(*bug_blaster['pos'])
        
        centipedes = self.game_state.get('centipedes', [])
        
        # Check if any centipede segment is adjacent
        for centipede in centipedes:
            for segment in centipede['body']:
                seg_pos = Position(*segment)
                if my_pos.manhattan_distance(seg_pos) <= 1:
                    # Immediate threat - find any safe move
                    return self.find_safe_move()
        
        return None
    
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
        8. Center horizontally
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
            return action
        
        # 2. Horizontal trap escape
        if action := self.detect_horizontal_trap():
            self.debug_info['reason'] = 'horizontal_trap'
            return action
        
        # 3. Emergency evade
        if action := self.emergency_evade():
            self.debug_info['reason'] = 'emergency_evade'
            return action
        
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
                        return 'A'
                else:
                    # Não alinhados - tentar alinhar se for seguro
                    # Só alinhar se o alvo está a 3+ linhas acima (seguro)
                    if target_y < my_pos.y - 2:
                        if my_pos.x < target_x:
                            self.debug_info['reason'] = 'defensive_align_right'
                            move = self.find_safe_move('d')
                            if move == 'd':  # Só move se for seguro
                                return move
                        elif my_pos.x > target_x:
                            self.debug_info['reason'] = 'defensive_align_left'
                            move = self.find_safe_move('a')
                            if move == 'a':  # Só move se for seguro
                                return move
            
            # Mover para safe zone e evadir
            if my_pos.y < self.safe_zone_start:
                action = self.find_safe_move('s')
            else:
                action = self.find_safe_move()
            return action
        
        # 5. Aggressive shooting
        if self.current_strategy == "aggressive":
            target = self.find_best_target()
            
            if target:
                target_name, target_pos, target_score = target
                target_x, target_y = target_pos
                
                # Are we aligned?
                if my_pos.x == target_x:
                    # Check if path is clear
                    mushrooms = self.get_mushroom_positions()
                    path_clear = True
                    for y in range(target_y + 1, my_pos.y):
                        if (target_x, y) in mushrooms:
                            path_clear = False
                            break
                    
                    if path_clear and self.shot_cooldown == 0:
                        self.debug_info['reason'] = 'shooting_target'
                        self.shot_cooldown = 10
                        return 'A'
                    elif not path_clear and self.shot_cooldown == 0:
                        # Shoot mushroom blocking path
                        self.debug_info['reason'] = 'clearing_shot_path'
                        self.shot_cooldown = 10
                        return 'A'
                
                # Move towards target column
                if my_pos.x < target_x:
                    self.debug_info['reason'] = 'aligning_right'
                    return self.find_safe_move('d')
                elif my_pos.x > target_x:
                    self.debug_info['reason'] = 'aligning_left'
                    return self.find_safe_move('a')
        
        # 6. Clearing mushrooms
        if self.current_strategy == "clearing":
            self.debug_info['reason'] = 'clearing_mode'
            # Shoot upward to clear mushrooms
            if self.shot_cooldown == 0:
                self.shot_cooldown = 10
                return 'A'
            # Move around while waiting
            return self.find_safe_move()
        
        # 7. Return to home row (safe zone)
        if my_pos.y < self.safe_zone_start:
            self.debug_info['reason'] = 'return_to_safe_zone'
            return self.find_safe_move('s')
        
        # 8. Center horizontally
        center_x = self.map_size[0] // 2
        if abs(my_pos.x - center_x) > 3:
            if my_pos.x < center_x:
                self.debug_info['reason'] = 'centering_right'
                return self.find_safe_move('d')
            else:
                self.debug_info['reason'] = 'centering_left'
                return self.find_safe_move('a')
        
        # 9. Fallback
        self.debug_info['reason'] = 'fallback_safe'
        return self.find_safe_move()
    
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
        
        while True:
            try:
                # Receive game state
                state = json.loads(await websocket.recv())
                
                # Update agent state
                agent.update_state(state)
                
                # Decide action
                key = agent.decide_action()
                agent.last_action = key
                
                # Send action
                await websocket.send(json.dumps({"cmd": "key", "key": key}))
                
                # Debug output every 50 frames
                if agent.frame_count % 50 == 0:
                    print(f"Frame {agent.frame_count}: Strategy={agent.current_strategy}, "
                          f"Action={key}, Reason={agent.debug_info.get('reason', 'none')}, "
                          f"Score={state.get('score', 0)}")
                
            except websockets.exceptions.ConnectionClosedOK:
                print("Server disconnected")
                return


# Entry point
if __name__ == "__main__":
    SERVER = os.environ.get("SERVER", "localhost")
    PORT = os.environ.get("PORT", "8000")
    NAME = os.environ.get("NAME", getpass.getuser())
    asyncio.run(agent_loop(f"{SERVER}:{PORT}", NAME))