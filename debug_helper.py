#!/usr/bin/env python3
"""
Debug Helper Script for Centipede Agent
Analyzes agent_debug.log and provides insights
"""

import re
import sys
import io
from collections import Counter, defaultdict
from typing import List, Tuple


class LogAnalyzer:
    """Analyzes agent debug logs"""
    
    def __init__(self, log_file: str = "agent_debug.log"):
        self.log_file = log_file
        self.lines = []
        self.load_log()
    
    def load_log(self):
        """Load log file"""
        try:
            # Open the log file using UTF-8 and replace any invalid bytes
            with open(self.log_file, 'r', encoding='utf-8', errors='replace') as f:
                self.lines = f.readlines()
            print(f"✓ Loaded {len(self.lines)} log lines from {self.log_file}")
        except FileNotFoundError:
            print(f"✗ Log file {self.log_file} not found!")
            sys.exit(1)
    
    def analyze_strategies(self):
        """Analyze strategy changes"""
        print("\n" + "="*60)
        print("STRATEGY ANALYSIS")
        print("="*60)
        
        strategies = []
        for line in self.lines:
            if "Switching to" in line:
                match = re.search(r'Switching to (\w+) strategy', line)
                if match:
                    strategies.append(match.group(1))
        
        if strategies:
            strategy_counts = Counter(strategies)
            print(f"\nTotal strategy changes: {len(strategies)}")
            print("\nStrategy distribution:")
            for strategy, count in strategy_counts.most_common():
                print(f"  {strategy:12} : {count:3} times ({count/len(strategies)*100:.1f}%)")
            
            # Analyze patterns
            print(f"\nStrategy sequence: {' → '.join(strategies[:10])}...")
        else:
            print("No strategy changes found in log")
    
    def analyze_performance(self):
        """Analyze performance metrics"""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        hits = []
        shots = []
        accuracy = []
        scores = []
        
        for line in self.lines:
            if "Performance:" in line:
                # Extract stats
                hit_match = re.search(r"'hits_made': (\d+)", line)
                shot_match = re.search(r"'shots_fired': (\d+)", line)
                acc_match = re.search(r"'accuracy': ([\d.]+)", line)
                score_match = re.search(r"'score_per_step': ([\d.]+)", line)
                
                if hit_match:
                    hits.append(int(hit_match.group(1)))
                if shot_match:
                    shots.append(int(shot_match.group(1)))
                if acc_match:
                    accuracy.append(float(acc_match.group(1)))
                if score_match:
                    scores.append(float(score_match.group(1)))
        
        if hits:
            print(f"\nHits made: {max(hits)} total")
            print(f"Shots fired: {max(shots)} total")
            print(f"Final accuracy: {accuracy[-1]:.2f}%" if accuracy else "N/A")
            print(f"Score per step: {scores[-1]:.3f}" if scores else "N/A")
            
            if accuracy:
                avg_acc = sum(accuracy) / len(accuracy)
                print(f"Average accuracy: {avg_acc:.2f}%")
                
                if avg_acc < 20:
                    print("  ⚠ WARNING: Low accuracy - agent shooting randomly")
                elif avg_acc > 30:
                    print("  ✓ GOOD: High accuracy - well-positioned shots")
        else:
            print("No performance data found in log")
    
    def analyze_threats(self):
        """Analyze threat events"""
        print("\n" + "="*60)
        print("THREAT ANALYSIS")
        print("="*60)
        
        emergencies = 0
        high_threats = 0
        stuck_centipedes = []
        
        for line in self.lines:
            if "EMERGENCY!" in line:
                emergencies += 1
            if "High threat" in line:
                high_threats += 1
            if "appears stuck" in line:
                match = re.search(r'Centipede (\S+) appears stuck', line)
                if match:
                    stuck_centipedes.append(match.group(1))
        
        print(f"\nEmergency evacuations: {emergencies}")
        print(f"High threat warnings: {high_threats}")
        
        if stuck_centipedes:
            print(f"\nStuck centipedes detected: {len(set(stuck_centipedes))}")
            print(f"  Unique: {', '.join(set(stuck_centipedes))}")
        
        if emergencies > 50:
            print("  ⚠ WARNING: Too many emergencies - agent struggling to survive")
        elif emergencies < 10:
            print("  ✓ GOOD: Low emergency count - good threat avoidance")
    
    def analyze_shots(self):
        """Analyze shooting behavior"""
        print("\n" + "="*60)
        print("SHOOTING ANALYSIS")
        print("="*60)
        
        shot_reasons = []
        hit_count = 0
        
        for line in self.lines:
            if "Shooting:" in line:
                match = re.search(r'Shooting: (.+)$', line)
                if match:
                    shot_reasons.append(match.group(1))
            if "HIT!" in line:
                hit_count += 1
        
        if shot_reasons:
            print(f"\nTotal shots taken: {len(shot_reasons)}")
            print(f"Confirmed hits: {hit_count}")
            
            # Categorize shot reasons
            good_shots = sum(1 for r in shot_reasons if "Good shot" in r)
            predictive_shots = sum(1 for r in shot_reasons if "Predictive" in r)
            
            print(f"\nShot types:")
            print(f"  Good shots: {good_shots}")
            print(f"  Predictive shots: {predictive_shots}")
            
            # Sample some shot reasons
            print(f"\nSample shot reasons:")
            for reason in shot_reasons[:5]:
                print(f"  - {reason}")
        else:
            print("No shooting data found in log")
    
    def find_death_cause(self):
        """Try to determine cause of death"""
        print("\n" + "="*60)
        print("DEATH ANALYSIS")
        print("="*60)
        
        # Look for events near the end of log
        last_lines = self.lines[-50:]
        
        emergencies_at_end = sum(1 for line in last_lines if "EMERGENCY" in line)
        high_threats_at_end = sum(1 for line in last_lines if "High threat" in line)
        
        print(f"\nIn final 50 log entries:")
        print(f"  Emergencies: {emergencies_at_end}")
        print(f"  High threats: {high_threats_at_end}")
        
        if emergencies_at_end > 10:
            print("\n  ⚠ LIKELY CAUSE: Overwhelmed by threats, couldn't escape")
        elif emergencies_at_end == 0 and high_threats_at_end == 0:
            print("\n  ? UNCLEAR: Death was sudden, possibly direct collision")
        
        # Show last few significant events
        print("\nLast significant events:")
        significant = []
        for line in last_lines:
            if any(keyword in line for keyword in ["EMERGENCY", "High threat", "HIT", "Switching"]):
                significant.append(line.strip())
        
        for event in significant[-10:]:
            # Extract just the message part
            parts = event.split(" - ")
            if len(parts) >= 4:
                print(f"  {parts[-1]}")
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Analyze patterns and suggest fixes
        for line in self.lines:
            if "Low accuracy" in line:
                recommendations.append(
                    "• Lower shooting threshold (currently 30) to increase shots"
                )
            if "Too many emergencies" in line:
                recommendations.append(
                    "• Increase danger zone radius for earlier threat detection"
                )
        
        # Count strategy uses
        defensive_count = sum(1 for line in self.lines if "DEFENSIVE strategy" in line)
        aggressive_count = sum(1 for line in self.lines if "AGGRESSIVE strategy" in line)
        
        if defensive_count > aggressive_count * 2:
            recommendations.append(
                "• Agent is too defensive - consider lowering threat_count threshold"
            )
        elif aggressive_count > defensive_count * 3:
            recommendations.append(
                "• Agent may be too aggressive - increase caution in threatening situations"
            )
        
        # Check stuck centipedes
        stuck_count = sum(1 for line in self.lines if "appears stuck" in line)
        if stuck_count > 5:
            recommendations.append(
                "• Multiple stuck centipedes - may need better mushroom clearing strategy"
            )
        
        if recommendations:
            print("\nBased on log analysis:")
            for rec in recommendations:
                print(f"{rec}")
        else:
            print("\nNo specific recommendations - performance looks reasonable!")
    
    def full_analysis(self):
        """Run complete analysis"""
        print("\n" + "╔" + "="*58 + "╗")
        print("║" + " "*15 + "CENTIPEDE AGENT LOG ANALYZER" + " "*15 + "║")
        print("╚" + "="*58 + "╝")
        
        self.analyze_performance()
        self.analyze_strategies()
        self.analyze_threats()
        self.analyze_shots()
        self.find_death_cause()
        self.generate_recommendations()
        
        print("\n" + "="*60)
        print("Analysis complete!")
        print("="*60 + "\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Centipede agent debug logs")
    parser.add_argument(
        '--log',
        default='agent_debug.log',
        help='Path to log file (default: agent_debug.log)'
    )
    parser.add_argument(
        '--section',
        choices=['performance', 'strategies', 'threats', 'shots', 'death', 'all'],
        default='all',
        help='Specific section to analyze'
    )
    
    args = parser.parse_args()
    # Prepare for running as script: ensure stdout uses UTF-8 when redirected
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

    analyzer = LogAnalyzer(args.log)

    if args.section == 'all':
        analyzer.full_analysis()
    elif args.section == 'performance':
        analyzer.analyze_performance()
    elif args.section == 'strategies':
        analyzer.analyze_strategies()
    elif args.section == 'threats':
        analyzer.analyze_threats()
    elif args.section == 'shots':
        analyzer.analyze_shots()
    elif args.section == 'death':
        analyzer.find_death_cause()


if __name__ == "__main__":
    main()