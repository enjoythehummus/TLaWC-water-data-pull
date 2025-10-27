#!/usr/bin/env python3
"""
Water Health Analysis Tool
Analyzes water monitoring data and provides health grades based on historical trends
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class WaterHealthAnalyzer:
    """Analyzes water quality parameters and generates health scores"""

    # Water quality parameter thresholds and ideal ranges (based on general water health standards)
    PARAMETER_STANDARDS = {
        'Water Temperature': {
            'ideal_range': (10, 25),  # �C - optimal for most aquatic life
            'acceptable_range': (5, 30),
            'critical_high': 35,
            'critical_low': 0,
            'unit': '�C',
            'higher_is_worse': False  # Temperature extremes (both high and low) are bad
        },
        'Water Course Level': {
            'use_percentiles': True,  # Use historical percentiles instead of fixed ranges
            'unit': 'm',
            'higher_is_worse': False
        },
        'Water Course Discharge': {
            'use_percentiles': True,  # Flow varies significantly by location
            'unit': 'm�/s',
            'higher_is_worse': False
        },
        'Turbidity': {
            'ideal_range': (0, 25),  # NTU - low turbidity indicates clear water
            'acceptable_range': (0, 50),
            'critical_high': 100,
            'unit': 'NTU',
            'higher_is_worse': True
        },
        'Rainfall': {
            'use_percentiles': True,
            'unit': 'mm',
            'higher_is_worse': False
        },
        'Storage Level': {
            'use_percentiles': True,
            'unit': 'm',
            'higher_is_worse': False
        },
        'Storage Volume': {
            'use_percentiles': True,
            'unit': 'ML',
            'higher_is_worse': False
        }
    }

    def __init__(self, historical_data_dir: str):
        """
        Initialize the analyzer

        Args:
            historical_data_dir: Path to historical_data directory
        """
        self.historical_data_dir = historical_data_dir

    def load_parameter_data(self, station_id: str, parameter: str) -> Optional[Dict]:
        """
        Load historical data for a specific parameter

        Args:
            station_id: Station ID
            parameter: Parameter name

        Returns:
            Dictionary with parameter data or None if not found
        """
        # Sanitize parameter name for filename
        filename = parameter.replace(' ', '_').replace('/', '_').replace('\\', '_') + '.json'
        filepath = os.path.join(self.historical_data_dir, station_id, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def calculate_trend(self, timestamps: List[str], values: List[float],
                       recent_days: int = 30) -> Dict:
        """
        Calculate trend over recent period

        Args:
            timestamps: List of ISO timestamp strings
            values: List of measurement values
            recent_days: Number of recent days to analyze

        Returns:
            Dictionary with trend information
        """
        if len(timestamps) < 2 or len(values) < 2:
            return {'trend': 'insufficient_data', 'slope': 0, 'r_squared': 0}

        # Parse timestamps and filter for recent data
        cutoff_date = datetime.now() - timedelta(days=recent_days)
        recent_data = []

        for ts_str, val in zip(timestamps, values):
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                if dt >= cutoff_date:
                    recent_data.append((dt.timestamp(), val))
            except:
                continue

        if len(recent_data) < 2:
            return {'trend': 'insufficient_recent_data', 'slope': 0, 'r_squared': 0}

        # Linear regression using numpy
        times = np.array([d[0] for d in recent_data])
        vals = np.array([d[1] for d in recent_data])

        # Calculate slope and r-squared using numpy
        coefficients = np.polyfit(times, vals, 1)
        slope = coefficients[0]

        # Calculate r-squared
        p = np.poly1d(coefficients)
        yhat = p(times)
        ybar = np.mean(vals)
        ssreg = np.sum((yhat - ybar) ** 2)
        sstot = np.sum((vals - ybar) ** 2)
        r_squared = ssreg / sstot if sstot > 0 else 0

        # Calculate correlation for p-value approximation
        correlation = np.corrcoef(times, vals)[0, 1]
        p_value = 0.05 if abs(correlation) > 0.5 else 0.1  # Simplified p-value

        # Classify trend
        if abs(slope) < 0.0001:  # Essentially flat
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'

        return {
            'trend': trend,
            'slope': float(slope),
            'r_squared': float(r_squared),
            'p_value': float(p_value)
        }

    def calculate_health_score(self, parameter: str, current_value: float,
                               historical_stats: Dict, trend_info: Dict) -> Dict:
        """
        Calculate health score (0-100) for a parameter

        Args:
            parameter: Parameter name
            current_value: Current measurement value
            historical_stats: Historical statistics dictionary
            trend_info: Trend analysis dictionary

        Returns:
            Dictionary with score and explanation
        """
        score = 50  # Start at neutral
        factors = []

        standards = self.PARAMETER_STANDARDS.get(parameter, {})

        # Score based on fixed ranges (if available)
        if 'ideal_range' in standards:
            ideal_min, ideal_max = standards['ideal_range']
            acceptable_min, acceptable_max = standards['acceptable_range']
            critical_high = standards.get('critical_high')
            critical_low = standards.get('critical_low')

            if ideal_min <= current_value <= ideal_max:
                score = 95
                factors.append('Within ideal range')
            elif acceptable_min <= current_value <= acceptable_max:
                score = 75
                factors.append('Within acceptable range')
            elif critical_high and current_value > critical_high:
                score = 20
                factors.append('Critically high')
            elif critical_low is not None and current_value < critical_low:
                score = 20
                factors.append('Critically low')
            else:
                score = 50
                factors.append('Outside ideal range')

        # Score based on historical percentiles
        elif standards.get('use_percentiles') and historical_stats:
            p25 = historical_stats.get('percentile_25', 0)
            p50 = historical_stats.get('median', 0)
            p75 = historical_stats.get('percentile_75', 0)
            p95 = historical_stats.get('percentile_95', 0)

            # Check if current value is within normal range (25th-75th percentile)
            if p25 <= current_value <= p75:
                score = 85
                factors.append('Within normal historical range (25th-75th percentile)')
            elif p50 * 0.5 <= current_value <= p95:
                score = 70
                factors.append('Within extended historical range')
            elif current_value > p95:
                score = 50
                factors.append('Above 95th percentile (unusually high)')
            elif current_value < p25:
                score = 50
                factors.append('Below 25th percentile (unusually low)')

        # Adjust based on trend
        if trend_info['trend'] == 'stable':
            score += 5
            factors.append('Stable trend')
        elif trend_info['trend'] == 'increasing':
            if standards.get('higher_is_worse'):
                score -= 10
                factors.append('Increasing trend (concerning)')
            else:
                factors.append('Increasing trend')
        elif trend_info['trend'] == 'decreasing':
            if standards.get('higher_is_worse'):
                score += 5
                factors.append('Decreasing trend (improving)')
            else:
                score -= 5
                factors.append('Decreasing trend')

        # Ensure score is within bounds
        score = max(0, min(100, score))

        # Assign grade
        if score >= 90:
            grade = 'A'
            health_status = 'Excellent'
        elif score >= 80:
            grade = 'B'
            health_status = 'Good'
        elif score >= 70:
            grade = 'C'
            health_status = 'Fair'
        elif score >= 60:
            grade = 'D'
            health_status = 'Poor'
        else:
            grade = 'F'
            health_status = 'Critical'

        return {
            'score': round(score, 1),
            'grade': grade,
            'health_status': health_status,
            'factors': factors,
            'current_value': current_value,
            'unit': standards.get('unit', '')
        }

    def analyze_station(self, station_id: str, station_name: str,
                       latest_snapshot: Dict) -> Dict:
        """
        Analyze all parameters for a station

        Args:
            station_id: Station ID
            station_name: Station name
            latest_snapshot: Latest data snapshot

        Returns:
            Analysis results dictionary
        """
        results = {
            'station_id': station_id,
            'station_name': station_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'parameters': {},
            'overall_score': 0,
            'overall_grade': 'N/A'
        }

        parameter_scores = []

        for param_entry in latest_snapshot:
            if param_entry['site_id'] != station_id:
                continue

            parameter = param_entry['data_type']
            current_value = param_entry['value']

            if current_value is None:
                continue

            try:
                current_value = float(current_value)
            except:
                continue

            # Load historical data
            historical_data = self.load_parameter_data(station_id, parameter)

            if not historical_data:
                continue

            # Extract measurements
            measurements = historical_data.get('measurements', [])
            if not measurements:
                continue

            timestamps = [m['timestamp'] for m in measurements]
            values = [float(m['value']) for m in measurements if m['value'] is not None]

            # Calculate trend
            trend_info = self.calculate_trend(timestamps, values)

            # Calculate health score
            health_info = self.calculate_health_score(
                parameter,
                current_value,
                historical_data.get('statistics', {}),
                trend_info
            )

            # Add historical context
            health_info['historical_stats'] = {
                'mean': historical_data['statistics'].get('mean'),
                'median': historical_data['statistics'].get('median'),
                'min': historical_data['statistics'].get('min'),
                'max': historical_data['statistics'].get('max'),
                'data_range': historical_data.get('data_range')
            }
            health_info['trend'] = trend_info

            results['parameters'][parameter] = health_info
            parameter_scores.append(health_info['score'])

        # Calculate overall station score
        if parameter_scores:
            overall_score = np.mean(parameter_scores)
            results['overall_score'] = round(overall_score, 1)

            if overall_score >= 90:
                results['overall_grade'] = 'A'
            elif overall_score >= 80:
                results['overall_grade'] = 'B'
            elif overall_score >= 70:
                results['overall_grade'] = 'C'
            elif overall_score >= 60:
                results['overall_grade'] = 'D'
            else:
                results['overall_grade'] = 'F'

        return results

    def analyze_all_stations(self, latest_snapshot_file: str,
                            station_names: Dict[str, str]) -> List[Dict]:
        """
        Analyze all stations from latest snapshot

        Args:
            latest_snapshot_file: Path to latest snapshot file
            station_names: Dictionary mapping station IDs to names

        Returns:
            List of analysis results for each station
        """
        # Load latest snapshot
        with open(latest_snapshot_file, 'r') as f:
            snapshot = json.load(f)

        # Group by station
        stations = {}
        for entry in snapshot:
            station_id = entry['site_id']
            if station_id not in stations:
                stations[station_id] = []
            stations[station_id].append(entry)

        # Analyze each station
        results = []
        for station_id, station_data in stations.items():
            station_name = station_names.get(station_id, f"Station {station_id}")
            analysis = self.analyze_station(station_id, station_name, station_data)
            results.append(analysis)

        return results

    def save_analysis(self, results: List[Dict], output_file: str):
        """
        Save analysis results to JSON file

        Args:
            results: List of analysis results
            output_file: Output file path
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nAnalysis saved to {output_file}")

    def print_summary(self, results: List[Dict]):
        """
        Print summary of analysis results

        Args:
            results: List of analysis results
        """
        print(f"\n{'='*80}")
        print(f"WATER HEALTH ANALYSIS SUMMARY")
        print(f"{'='*80}\n")

        for station in sorted(results, key=lambda x: x['overall_score'], reverse=True):
            print(f"Station: {station['station_name']} ({station['station_id']})")
            print(f"Overall Score: {station['overall_score']}/100 (Grade: {station['overall_grade']})")
            print(f"Parameters Analyzed: {len(station['parameters'])}\n")

            for param_name, param_info in station['parameters'].items():
                print(f"  {param_name}:")
                print(f"    Current: {param_info['current_value']} {param_info['unit']}")
                print(f"    Score: {param_info['score']}/100 (Grade: {param_info['grade']}) - {param_info['health_status']}")
                print(f"    Trend: {param_info['trend']['trend']}")
                if param_info['factors']:
                    print(f"    Factors: {', '.join(param_info['factors'])}")
                print()

            print(f"{'-'*80}\n")


def main():
    """Main function to run water health analysis"""

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load station names
    station_file = os.path.join(script_dir, 'site_list_test.json')
    with open(station_file, 'r') as f:
        site_data = json.load(f)

    station_names = {}
    for site in site_data.get('sites', []):
        if site.get('site_status', '').lower() == 'active':
            station_names[site['site_id']] = site['site_name']

    # Find most recent snapshot
    snapshot_dir = os.path.join(script_dir, 'data_snapshot_simplified')
    snapshot_files = [f for f in os.listdir(snapshot_dir) if f.startswith('bom_water_latest_')]

    if not snapshot_files:
        print("No snapshot files found!")
        return

    # Use most recent snapshot
    latest_snapshot = os.path.join(snapshot_dir, sorted(snapshot_files)[-1])
    print(f"Analyzing snapshot: {latest_snapshot}")

    # Initialize analyzer
    historical_data_dir = os.path.join(script_dir, 'historical_data')
    analyzer = WaterHealthAnalyzer(historical_data_dir)

    # Run analysis
    results = analyzer.analyze_all_stations(latest_snapshot, station_names)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(script_dir, f'analysis_results/water_health_analysis_{timestamp}.json')
    analyzer.save_analysis(results, output_file)

    # Print summary
    analyzer.print_summary(results)

    print(f"{'='*80}")
    print(f"Analysis complete! {len(results)} stations analyzed.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
