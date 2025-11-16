"""
Advanced Analysis System for Centipede Agent
Handles trajectory prediction, pattern detection, and strategic planning
"""

from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class ThreatAssessment:
    """Assessment of threat level for a position"""
    position: Tuple[int, int]
    threat_level: float  # 0.0 = safe, 1.0 = certain death
    threat_sources: List[str]  # What's causing the threat
    recommended_action: str  # Suggested action to take


class CentipedePredictor:
    """Predicts centipede movement patterns"""
    
    def __init__(self, map_width: int = 40, map_height: int = 24):
        self.map_width = map_width
        self.map_height = map_height
    
    def predict_trajectory(
        self,
        centipede_body: List[Tuple[int, int]],
        direction: int,
        mushrooms: Set[Tuple[int, int]],
        num_steps: int = 10
    ) -> List[List[Tuple[int, int]]]:
        """
        Predict centipede positions for next N steps
        Returns list of predicted body positions for each step
        """
        if not centipede_body:
            return []
        
        predictions = []
        current_body = centipede_body.copy()
        current_dir = direction
        move_dir = 1  # 1 = down, -1 = up
        waiting_to_move_vertically = False
        
        for step in range(num_steps):
            head = current_body[-1]
            
            # Calculate next position based on direction
            if current_dir == 1:  # EAST
                next_head = (head[0] + 1, head[1])
            elif current_dir == 3:  # WEST
                next_head = (head[0] - 1, head[1])
            else:
                # If moving vertically, continue current horizontal direction
                next_head = head
            
            # Check for wall or mushroom collision
            if (next_head[0] < 0 or next_head[0] >= self.map_width or 
                next_head in mushrooms):
                
                # Need to move vertically and change direction
                if head[1] == 0:
                    move_dir = 1
                elif head[1] >= self.map_height - 1:
                    move_dir = -1
                
                # Try vertical move
                next_head_vert = (head[0], head[1] + move_dir)
                
                if (0 <= next_head_vert[1] < self.map_height and 
                    next_head_vert not in mushrooms):
                    next_head = next_head_vert
                    waiting_to_move_vertically = False
                else:
                    next_head = head
                    waiting_to_move_vertically = True
                
                # Reverse direction
                current_dir = 1 if current_dir == 3 else 3
            
            # Resolve vertical debt
            if waiting_to_move_vertically:
                next_head_vert = (head[0], head[1] + move_dir)
                if (0 <= next_head_vert[1] < self.map_height and 
                    next_head_vert not in mushrooms):
                    next_head = next_head_vert
                    waiting_to_move_vertically = False
            
            # Update body
            current_body.append(next_head)
            current_body.pop(0)
            
            predictions.append(current_body.copy())
        
        return predictions
    
    def detect_stuck_centipede(
        self,
        centipede_body: List[Tuple[int, int]],
        history: List[Tuple[int, int]],
        threshold: int = 10
    ) -> bool:
        """
        Detect if centipede is stuck (not moving properly)
        Returns True if centipede appears stuck
        """
        if len(history) < threshold:
            return False
        
        # Check if head position has changed very little
        recent_positions = history[-threshold:]
        unique_positions = len(set(recent_positions))
        
        # If very few unique positions, likely stuck
        return unique_positions < threshold // 2
    
    def find_safe_shooting_lanes(
        self,
        blaster_pos: Tuple[int, int],
        centipede_predictions: List[List[List[Tuple[int, int]]]],  # List of centipede predictions
        mushrooms: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int, float]]:  # (x, y, hit_probability)
        """
        Find positions where we're likely to hit centipede
        Returns list of (x, y, probability) tuples
        centipede_predictions: List of predictions for each centipede
        """
        lanes = []
        blaster_x, blaster_y = blaster_pos
        
        # For each column
        for x in range(self.map_width):
            # Count how many predicted positions are in this column
            hits = 0
            total_predictions = 0
            min_y = blaster_y
            
            # Iterate through all centipedes
            for centipede_pred in centipede_predictions:
                # Each centipede_pred is a list of future body states
                for future_body in centipede_pred:
                    total_predictions += 1
                    # future_body is a list of (x, y) tuples representing segments
                    # Validate that future_body is iterable and contains tuples
                    if not isinstance(future_body, (list, tuple)):
                        continue
                    
                    for segment in future_body:
                        # Ensure segment is a tuple/list with at least 2 elements
                        if not isinstance(segment, (list, tuple)) or len(segment) < 2:
                            continue
                        
                        seg_x, seg_y = segment[0], segment[1]
                        if seg_x == x and seg_y < blaster_y:
                            hits += 1
                            min_y = min(min_y, seg_y)
            
            if total_predictions > 0:
                probability = hits / total_predictions
                
                # Check if path is clear from min_y to blaster
                clear = True
                if min_y < blaster_y:
                    for y in range(min_y, blaster_y):
                        if (x, y) in mushrooms:
                            clear = False
                            break
                
                if clear and probability > 0.05:  # Lower threshold for better detection
                    lanes.append((x, blaster_y, probability))
        
        # Sort by probability
        lanes.sort(key=lambda l: l[2], reverse=True)
        return lanes


class MushroomAnalyzer:
    """Analyzes mushroom patterns and suggests clearing strategies"""
    
    def __init__(self, map_width: int = 40, map_height: int = 24):
        self.map_width = map_width
        self.map_height = map_height
    
    def find_densest_areas(
        self,
        mushrooms: Set[Tuple[int, int]],
        grid_size: int = 5
    ) -> List[Tuple[int, int, int]]:  # (x, y, density)
        """
        Find areas with highest mushroom density
        Returns list of (center_x, center_y, count)
        """
        density_map = []
        
        for center_x in range(0, self.map_width, grid_size):
            for center_y in range(0, self.map_height, grid_size):
                count = sum(
                    1 for mx, my in mushrooms
                    if (center_x <= mx < center_x + grid_size and
                        center_y <= my < center_y + grid_size)
                )
                
                if count > 0:
                    density_map.append((center_x, center_y, count))
        
        density_map.sort(key=lambda d: d[2], reverse=True)
        return density_map
    
    def suggest_tunnel_positions(
        self,
        mushrooms: Set[Tuple[int, int]],
        num_tunnels: int = 2
    ) -> List[int]:  # List of x coordinates for tunnels
        """
        Suggest X coordinates where tunnels should be created
        Strategy: Create clear vertical lanes for centipedes to fall through
        """
        # Count mushrooms per column
        column_counts = {}
        for x in range(self.map_width):
            count = sum(1 for mx, my in mushrooms if mx == x)
            column_counts[x] = count
        
        # Find columns with fewest mushrooms (easier to clear)
        sorted_columns = sorted(column_counts.items(), key=lambda c: c[1])
        
        # Select well-spaced tunnel positions
        tunnels = []
        spacing = self.map_width // (num_tunnels + 1)
        
        for i in range(num_tunnels):
            target_x = spacing * (i + 1)
            # Find closest column with low mushroom count
            closest = min(
                sorted_columns[:self.map_width // 2],
                key=lambda c: abs(c[0] - target_x)
            )
            tunnels.append(closest[0])
        
        return tunnels
    
    def calculate_clear_priority(
        self,
        mushroom: Tuple[int, int],
        blaster_pos: Tuple[int, int],
        centipede_positions: List[Tuple[int, int]]
    ) -> float:
        """
        Calculate priority for clearing a specific mushroom
        Higher value = higher priority
        """
        mx, my = mushroom
        bx, by = blaster_pos
        
        priority = 0.0
        
        # Distance factor (closer = higher priority)
        distance = abs(mx - bx) + abs(my - by)
        priority += 50.0 / (distance + 1)
        
        # Strategic position factor
        # Mushrooms in our movement area are high priority
        if my >= by - 3:
            priority += 30.0
        
        # Mushrooms in shooting lanes are high priority
        if mx == bx:
            priority += 40.0
        
        # Mushrooms near centipede paths are high priority
        for cx, cy in centipede_positions:
            if abs(mx - cx) <= 2 and abs(my - cy) <= 2:
                priority += 20.0
                break
        
        return priority


class ThreatAnalyzer:
    """Analyzes threats and suggests evasive maneuvers"""
    
    def __init__(self, map_width: int = 40, map_height: int = 24):
        self.map_width = map_width
        self.map_height = map_height
    
    def assess_position(
        self,
        pos: Tuple[int, int],
        centipede_bodies: List[List[Tuple[int, int]]],
        centipede_predictions: List[List[List[Tuple[int, int]]]],
        mushrooms: Set[Tuple[int, int]]
    ) -> ThreatAssessment:
        """
        Comprehensive threat assessment for a position
        """
        x, y = pos
        threat_level = 0.0
        threat_sources = []
        
        # Check immediate centipede collision
        for body in centipede_bodies:
            if pos in body:
                threat_level = 1.0
                threat_sources.append("Direct centipede contact")
                return ThreatAssessment(
                    position=pos,
                    threat_level=threat_level,
                    threat_sources=threat_sources,
                    recommended_action="EMERGENCY_ESCAPE"
                )
        
        # Check proximity to centipedes
        min_distance = float('inf')
        for body in centipede_bodies:
            for seg in body:
                dist = abs(x - seg[0]) + abs(y - seg[1])
                min_distance = min(min_distance, dist)
        
        if min_distance <= 2:
            threat_level += 0.5
            threat_sources.append(f"Centipede within {min_distance} tiles")
        
        # Check predicted centipede paths
        for predictions in centipede_predictions:
            for future_body in predictions[:5]:  # Check next 5 steps
                if pos in future_body:
                    threat_level += 0.3
                    threat_sources.append("In predicted centipede path")
                    break
        
        # Check if trapped (surrounded by mushrooms)
        surrounding_mushrooms = sum(
            1 for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
            if (x+dx, y+dy) in mushrooms
        )
        if surrounding_mushrooms >= 3:
            threat_level += 0.4
            threat_sources.append("Surrounded by mushrooms")
        
        # Determine recommended action
        if threat_level >= 0.8:
            recommended_action = "EMERGENCY_ESCAPE"
        elif threat_level >= 0.5:
            recommended_action = "TACTICAL_RETREAT"
        elif threat_level >= 0.3:
            recommended_action = "STAY_ALERT"
        else:
            recommended_action = "OFFENSIVE"
        
        return ThreatAssessment(
            position=pos,
            threat_level=min(threat_level, 1.0),
            threat_sources=threat_sources,
            recommended_action=recommended_action
        )
    
    def find_safe_retreat_positions(
        self,
        current_pos: Tuple[int, int],
        danger_zones: Set[Tuple[int, int]],
        mushrooms: Set[Tuple[int, int]],
        max_distance: int = 5
    ) -> List[Tuple[Tuple[int, int], float]]:  # (position, safety_score)
        """
        Find safe positions to retreat to
        Returns list of (position, safety_score) tuples
        """
        cx, cy = current_pos
        safe_positions = []
        
        for dx in range(-max_distance, max_distance + 1):
            for dy in range(-max_distance, max_distance + 1):
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = cx + dx, cy + dy
                
                # Check boundaries
                if not (0 <= nx < self.map_width and 0 <= ny < self.map_height):
                    continue
                
                # Check if blocked by mushroom
                if (nx, ny) in mushrooms:
                    continue
                
                # Calculate safety score
                safety_score = 100.0
                
                # Distance from danger
                min_danger_dist = min(
                    (abs(nx - dx) + abs(ny - dy) for dx, dy in danger_zones),
                    default=999
                )
                safety_score += min_danger_dist * 10
                
                # Prefer positions not in corners
                corner_penalty = 0
                if nx < 3 or nx >= self.map_width - 3:
                    corner_penalty += 20
                if ny < 3 or ny >= self.map_height - 3:
                    corner_penalty += 20
                safety_score -= corner_penalty
                
                # Distance from current position (prefer closer)
                move_distance = abs(dx) + abs(dy)
                safety_score -= move_distance * 5
                
                safe_positions.append(((nx, ny), safety_score))
        
        safe_positions.sort(key=lambda p: p[1], reverse=True)
        return safe_positions