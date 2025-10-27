#!/usr/bin/env python3
"""
Bureau of Meteorology Water Data Fetcher
Retrieves water data from BOM's SOS2 service for specified monitoring stations
"""

import os
import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import schedule
import logging
from pathlib import Path


class BOMWaterDataFetcher:
    """Fetches water data from Bureau of Meteorology SOS2 service"""

    BASE_URL = "http://www.bom.gov.au/waterdata/services"
    LAST_FETCH_FILE = "last_fetch_timestamp.json"

    # XML namespaces used in SOS2 responses
    NAMESPACES = {
        'sos': 'http://www.opengis.net/sos/2.0',
        'om': 'http://www.opengis.net/om/2.0',
        'gml': 'http://www.opengis.net/gml/3.2',
        'wml2': 'http://www.opengis.net/waterml/2.0',
        'xlink': 'http://www.w3.org/1999/xlink',
        'gda': 'http://www.opengis.net/sosgda/1.0'
    }

    def __init__(self, station_ids: List[str]):
        """
        Initialize the fetcher with a list of station IDs

        Args:
            station_ids: List of BOM station IDs (e.g., ['403210', '403213'])
        """
        self.station_ids = station_ids
        self.data = {}

    @staticmethod
    def load_station_ids(filepath: str) -> List[str]:
        """
        Load station IDs from a JSON file (only active sites)

        Args:
            filepath: Path to JSON file structured like site_list_test.json

        Returns:
            List of station IDs for sites with "active" status
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        station_ids = []
        for site in data.get('sites', []):
            site_status = site.get('site_status', '').lower()
            site_id = site.get('site_id')

            # Only include sites with "active" status
            if site_id and site_status == 'active':
                station_ids.append(str(site_id))

        return station_ids

    @staticmethod
    def load_station_info(filepath: str) -> Dict[str, str]:
        """
        Load station ID to name mapping from a JSON file (only active sites)

        Args:
            filepath: Path to JSON file structured like site_list_test.json

        Returns:
            Dictionary mapping station IDs to station names
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        station_info = {}
        for site in data.get('sites', []):
            site_status = site.get('site_status', '').lower()
            site_id = site.get('site_id')
            site_name = site.get('site_name')

            # Only include sites with "active" status
            if site_id and site_status == 'active':
                station_info[str(site_id)] = site_name if site_name else f"Station {site_id}"

        return station_info

    def get_data_availability(self, station_id: str) -> Optional[Dict]:
        """
        Get available data for a station using GetDataAvailability request

        Args:
            station_id: BOM station ID

        Returns:
            Dictionary of available parameters and time series, or None if request fails
        """
        params = {
            'service': 'SOS',
            'version': '2.0',
            'request': 'GetDataAvailability',
            'featureOfInterest': f'http://bom.gov.au/waterdata/services/stations/{station_id}'
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)

            availability = {}
            for member in root.findall('.//gda:dataAvailabilityMember', self.NAMESPACES):
                # Extract parameter
                obs_prop = member.find('.//gda:observedProperty', self.NAMESPACES)
                if obs_prop is not None:
                    param_title = obs_prop.get('{http://www.w3.org/1999/xlink}title', 'Unknown')
                    param_href = obs_prop.get('{http://www.w3.org/1999/xlink}href', '')

                    # Extract procedure (time series type)
                    procedure = member.find('.//gda:procedure', self.NAMESPACES)
                    if procedure is not None:
                        proc_title = procedure.get('{http://www.w3.org/1999/xlink}title', '')
                        proc_href = procedure.get('{http://www.w3.org/1999/xlink}href', '')

                        # Extract time period
                        begin_pos = member.find('.//gml:beginPosition', self.NAMESPACES)
                        end_pos = member.find('.//gml:endPosition', self.NAMESPACES)

                        if param_title not in availability:
                            availability[param_title] = []

                        availability[param_title].append({
                            'parameter_href': param_href,
                            'procedure': proc_title,
                            'procedure_href': proc_href,
                            'begin_time': begin_pos.text if begin_pos is not None else None,
                            'end_time': end_pos.text if end_pos is not None else None
                        })

            return availability

        except Exception as e:
            print(f"Error getting data availability for station {station_id}: {e}")
            return None

    @staticmethod
    def get_most_recent_timestamp_for_station(station_dir: str) -> Optional[datetime]:
        """
        Get the most recent timestamp from all parameter files for a station

        Args:
            station_dir: Path to station directory (e.g., historical_data/403210/)

        Returns:
            datetime object of most recent entry, or None if no data exists
        """
        if not os.path.exists(station_dir):
            return None

        most_recent_dt = None

        try:
            # Check all JSON files in the station directory
            for filename in os.listdir(station_dir):
                if not filename.endswith('.json'):
                    continue

                filepath = os.path.join(station_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Get the end timestamp from data_range
                end_time_str = data.get('data_range', {}).get('end')
                if end_time_str:
                    try:
                        dt = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                        if most_recent_dt is None or dt > most_recent_dt:
                            most_recent_dt = dt
                    except:
                        continue

        except Exception as e:
            print(f"Error reading station directory {station_dir}: {e}")
            return None

        return most_recent_dt

    def get_latest_non_empty_observation(self, station_id: str, parameter_href: str,
                                         procedure_href: str, days_back: int = 90,
                                         start_date: Optional[datetime] = None) -> Optional[Dict]:
        """
        Get the most recent non-empty observation for a specific station, parameter, and time series type

        Args:
            station_id: BOM station ID
            parameter_href: Full parameter URL
            procedure_href: Full procedure (time series type) URL
            days_back: Number of days to look back for data (default: 90)
            start_date: Optional start date to fetch from (overrides days_back)

        Returns:
            Dictionary containing the latest non-empty observation data, or None if not found
        """
        from datetime import timedelta

        # Create temporal filter for recent data
        end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days_back)
        temporal_filter = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

        params = {
            'service': 'SOS',
            'version': '2.0',
            'request': 'GetObservation',
            'featureOfInterest': f'http://bom.gov.au/waterdata/services/stations/{station_id}',
            'observedProperty': parameter_href,
            'procedure': procedure_href,
            'temporalFilter': f'om:phenomenonTime,{temporal_filter}'
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)

            observations = []

            for obs_data in root.findall('.//sos:observationData', self.NAMESPACES):
                obs = obs_data.find('.//om:OM_Observation', self.NAMESPACES)
                if obs is None:
                    continue

                # Get phenomenon time
                phenom_time = obs.find('.//om:phenomenonTime', self.NAMESPACES)
                time_period = phenom_time.find('.//gml:TimePeriod', self.NAMESPACES)

                begin_time = None
                end_time = None
                if time_period is not None:
                    begin_elem = time_period.find('.//gml:beginPosition', self.NAMESPACES)
                    end_elem = time_period.find('.//gml:endPosition', self.NAMESPACES)
                    begin_time = begin_elem.text if begin_elem is not None else None
                    end_time = end_elem.text if end_elem is not None else None

                # Get result time
                result_time = obs.find('.//om:resultTime/gml:TimeInstant/gml:timePosition', self.NAMESPACES)
                result_time_str = result_time.text if result_time is not None else None

                # Get procedure and observed property
                procedure_elem = obs.find('.//om:procedure', self.NAMESPACES)
                obs_prop_elem = obs.find('.//om:observedProperty', self.NAMESPACES)

                procedure_name = procedure_elem.get('{http://www.w3.org/1999/xlink}title', '') if procedure_elem is not None else ''
                parameter_name = obs_prop_elem.get('{http://www.w3.org/1999/xlink}title', '') if obs_prop_elem is not None else ''

                # Get measurement data
                timeseries = obs.find('.//wml2:MeasurementTimeseries', self.NAMESPACES)
                if timeseries is None:
                    continue

                # Get default metadata
                default_meta = timeseries.find('.//wml2:DefaultTVPMeasurementMetadata', self.NAMESPACES)
                unit = None
                interpolation = None

                if default_meta is not None:
                    unit_elem = default_meta.find('.//wml2:uom', self.NAMESPACES)
                    interp_elem = default_meta.find('.//wml2:interpolationType', self.NAMESPACES)

                    unit = unit_elem.get('code', '') if unit_elem is not None else None
                    interpolation = interp_elem.get('{http://www.w3.org/1999/xlink}title', '') if interp_elem is not None else None

                # Get measurement points and find the latest non-empty value
                measurements = []
                latest_non_empty = None

                for point in timeseries.findall('.//wml2:point', self.NAMESPACES):
                    tvp = point.find('.//wml2:MeasurementTVP', self.NAMESPACES)
                    if tvp is None:
                        continue

                    time_elem = tvp.find('.//wml2:time', self.NAMESPACES)
                    value_elem = tvp.find('.//wml2:value', self.NAMESPACES)

                    # Get metadata (quality code, etc.)
                    metadata = tvp.find('.//wml2:metadata', self.NAMESPACES)
                    quality_code = None

                    if metadata is not None:
                        meta_data = metadata.find('.//wml2:DefaultTVPMeasurementMetadata', self.NAMESPACES)
                        if meta_data is not None:
                            qualifier = meta_data.find('.//wml2:qualifier', self.NAMESPACES)
                            if qualifier is not None:
                                quality_code = qualifier.get('{http://www.w3.org/1999/xlink}title', '')

                    # Check if value is not nil/empty
                    is_nil = value_elem.get('{http://www.w3.org/2001/XMLSchema-instance}nil') if value_elem is not None else True
                    value_text = value_elem.text if value_elem is not None and not is_nil else None

                    measurement = {
                        'time': time_elem.text if time_elem is not None else None,
                        'value': value_text,
                        'quality_code': quality_code
                    }
                    measurements.append(measurement)

                    # Track the latest non-empty measurement
                    if value_text is not None:
                        latest_non_empty = measurement

                # Only include observation if we found at least one non-empty value
                if latest_non_empty:
                    observation = {
                        'parameter': parameter_name,
                        'procedure': procedure_name,
                        'phenomenon_time': {
                            'begin': begin_time,
                            'end': end_time
                        },
                        'result_time': result_time_str,
                        'unit': unit,
                        'interpolation': interpolation,
                        'latest_value': latest_non_empty,  # Most recent non-empty value
                        'measurements': measurements,  # Store all measurements for merging
                        'total_measurements': len(measurements),
                        'non_empty_count': sum(1 for m in measurements if m['value'] is not None)
                    }

                    observations.append(observation)

            return {
                'station_id': station_id,
                'observations': observations
            }

        except Exception as e:
            print(f"Error getting observations for station {station_id}: {e}")
            return None

    @staticmethod
    def sanitize_parameter_name(parameter: str) -> str:
        """
        Convert parameter name to valid filename format

        Args:
            parameter: Parameter name (e.g., "Water Course Level")

        Returns:
            Sanitized filename (e.g., "Water_Course_Level")
        """
        return parameter.replace(' ', '_').replace('/', '_').replace('\\', '_')

    @staticmethod
    def merge_and_save_parameter_data(station_dir: str, station_id: str, observation: Dict,
                                      use_best_only: bool = False) -> Dict:
        """
        Merge new observation data with existing parameter file and save

        Args:
            station_dir: Path to station directory
            station_id: Station ID
            observation: Observation data dictionary
            use_best_only: If True, only use the latest/best data point instead of all measurements

        Returns:
            Dictionary with merge statistics
        """
        import numpy as np

        parameter = observation.get('parameter', 'Unknown')
        procedure = observation.get('procedure', 'Unknown')
        unit = observation.get('unit', '')
        interpolation = observation.get('interpolation', '')

        # Determine which measurements to use
        if use_best_only:
            # Use only the latest/best data point
            latest_value = observation.get('latest_value', {})
            if latest_value and latest_value.get('value') is not None:
                new_measurements = [latest_value]
            else:
                new_measurements = []
        else:
            # Use all measurements
            new_measurements = observation.get('measurements', [])

        # Create filename
        filename = BOMWaterDataFetcher.sanitize_parameter_name(parameter) + '.json'
        filepath = os.path.join(station_dir, filename)

        # Load existing data if it exists
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                historical_data = json.load(f)

            existing_measurements = historical_data.get('measurements', [])

            # Create set of existing timestamps
            existing_times = {m['timestamp'] for m in existing_measurements if m.get('timestamp')}

            # Add only new measurements
            new_count = 0
            for new_m in new_measurements:
                timestamp = new_m.get('time')
                if timestamp and timestamp not in existing_times:
                    existing_measurements.append({
                        'timestamp': timestamp,
                        'value': new_m.get('value'),
                        'quality_code': new_m.get('quality_code')
                    })
                    new_count += 1

            # Sort by timestamp
            existing_measurements.sort(key=lambda x: x.get('timestamp', ''))

            all_measurements = existing_measurements
            status = f"updated - added {new_count} new measurements"
        else:
            # Create new file
            all_measurements = [
                {
                    'timestamp': m.get('time'),
                    'value': m.get('value'),
                    'quality_code': m.get('quality_code')
                }
                for m in new_measurements
            ]
            all_measurements.sort(key=lambda x: x.get('timestamp', ''))
            status = f"created - {len(all_measurements)} measurements"

        # Calculate statistics
        values = [float(m['value']) for m in all_measurements if m.get('value') is not None]

        if values:
            statistics = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'percentile_25': float(np.percentile(values, 25)),
                'percentile_75': float(np.percentile(values, 75)),
                'percentile_95': float(np.percentile(values, 95)),
                'percentile_99': float(np.percentile(values, 99))
            }
        else:
            statistics = {}

        # Build data structure matching historical format
        data_to_save = {
            'station_id': station_id,
            'parameter': parameter,
            'procedure': procedure,
            'metadata': {
                'parameter': parameter,
                'procedure': procedure,
                'unit': unit,
                'interpolation': interpolation
            },
            'data_range': {
                'start': all_measurements[0]['timestamp'] if all_measurements else None,
                'end': all_measurements[-1]['timestamp'] if all_measurements else None
            },
            'statistics': statistics,
            'measurements': all_measurements
        }

        # Save to file
        os.makedirs(station_dir, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)

        return {
            'parameter': parameter,
            'unit': unit,
            'measurement_count': len(all_measurements),
            'statistics': statistics,
            'data_range': data_to_save['data_range'],
            'status': status
        }

    def fetch_all_latest_data(self, recommended_only: bool = True, start_date: Optional[datetime] = None,
                              full_data: bool = False) -> Dict:
        """
        Fetch the latest available data for all stations

        Args:
            recommended_only: If True, only fetch recommended time series types
            start_date: Optional start date to fetch from (if updating historical data)
            full_data: If True, fetch ALL time series for each parameter. If False, only best/recommended

        Returns:
            Dictionary containing all fetched data
        """
        start_time = time.time()

        # Recommended time series types (from Table 24 in documentation)
        recommended_procedures = [
            'DMQaQc.Merged.AsStored.1',
            'Pat4_C_B_1_PR01',  # Watercourse Discharge with flood warning data
            'PR02QaQc.Merged.AsStored.1',  # Storage Volume
            'PR02AHDQaQc.Merged.AsStored.1',  # Storage Level - AHD
        ]

        all_data = {
            'fetch_timestamp': datetime.now().isoformat(),
            'stations': {}
        }

        for i, station_id in enumerate(self.station_ids, 1):
            elapsed = time.time() - start_time
            print(f"Processing station {i}/{len(self.station_ids)}: {station_id} (Elapsed: {elapsed:.1f}s)")

            # Get available data for this station
            availability = self.get_data_availability(station_id)

            if availability is None:
                print(f"  No data availability information for station {station_id}")
                continue

            station_data = {
                'station_id': station_id,
                'data_availability': availability,
                'latest_observations': []
            }

            # Fetch observations for each available parameter
            for parameter, time_series_list in availability.items():
                print(f"  Parameter: {parameter} ({len(time_series_list)} time series available)")

                # Determine which time series to fetch
                time_series_to_fetch = []

                if full_data:
                    # For full snapshot, fetch ALL time series
                    time_series_to_fetch = time_series_list
                    print(f"    Fetching ALL {len(time_series_list)} time series (full data mode)")
                else:
                    # For simplified/summary snapshots and historical updates, fetch only the best one
                    best_ts = None

                    if recommended_only and time_series_list:
                        # Try to find a recommended time series
                        for ts in time_series_list:
                            procedure_name = ts['procedure']
                            is_recommended = any(rec in procedure_name for rec in recommended_procedures)
                            is_recommended = is_recommended or 'Merged' in procedure_name or 'AsStored' in procedure_name

                            if is_recommended:
                                best_ts = ts
                                break

                        # If no recommended found, use the first one
                        if best_ts is None:
                            best_ts = time_series_list[0]
                    else:
                        # If not filtering, just use the first time series
                        best_ts = time_series_list[0] if time_series_list else None

                    if best_ts:
                        time_series_to_fetch = [best_ts]

                # Fetch each selected time series
                for ts in time_series_to_fetch:
                    print(f"    Fetching: {ts['procedure']}")

                    # Fetch latest non-empty observation
                    # Use start_date if provided, otherwise default to 90 days back
                    obs_data = self.get_latest_non_empty_observation(
                        station_id,
                        ts['parameter_href'],
                        ts['procedure_href'],
                        days_back=90,
                        start_date=start_date
                    )

                    if obs_data and obs_data['observations']:
                        station_data['latest_observations'].extend(obs_data['observations'])
                        print(f"      ✓ Found {len(obs_data['observations'])} observations with data")
                    else:
                        print(f"      ✗ No data found")

                    # Add small delay to avoid overwhelming the server
                    time.sleep(0.5)

            all_data['stations'][station_id] = station_data

            # Add delay between stations
            time.sleep(1)

        total_time = time.time() - start_time
        all_data['total_fetch_time_seconds'] = round(total_time, 2)
        print(f"\nTotal fetch time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

        return all_data

    @staticmethod
    def create_station_summary(station_id: str, parameter_results: List[Dict]) -> Dict:
        """
        Create summary file for a station

        Args:
            station_id: Station ID
            parameter_results: List of parameter result dictionaries

        Returns:
            Summary dictionary
        """
        summary = {
            'station_id': station_id,
            'last_updated': datetime.now().isoformat(),
            'total_measurements': sum(p.get('measurement_count', 0) for p in parameter_results),
            'parameter_count': len(parameter_results),
            'parameters': parameter_results
        }

        return summary

    @staticmethod
    def save_station_summary(summary_dir: str, station_id: str, summary: Dict):
        """
        Save station summary to summaries directory

        Args:
            summary_dir: Path to summaries directory
            station_id: Station ID
            summary: Summary dictionary
        """
        os.makedirs(summary_dir, exist_ok=True)
        summary_file = os.path.join(summary_dir, f'{station_id}_summary.json')

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  Summary saved: {summary_file}")

    def save_to_json(self, data: Dict, filepath: str):
        """
        Save data to JSON file

        Args:
            data: Data dictionary to save
            filepath: Output file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSnapshot saved to {filepath}")

    def create_latest_snapshot(self, data: Dict, station_names: Dict[str, str]) -> List[Dict]:
        """
        Create snapshot with ONLY the most recent single data point per parameter

        Args:
            data: Full data dictionary from fetch
            station_names: Dictionary mapping station IDs to names

        Returns:
            List with only the latest single data point for each parameter
        """
        latest_only = []

        for station_id, station_data in data.get('stations', {}).items():
            # Get station name from the mapping
            station_name = station_names.get(station_id, f"Station {station_id}")

            for observation in station_data.get('latest_observations', []):
                latest_value = observation.get('latest_value', {})

                entry = {
                    'timestamp': latest_value.get('time'),
                    'site_id': station_id,
                    'site_name': station_name,
                    'data_type': observation.get('parameter'),
                    'value': latest_value.get('value'),
                    'unit': observation.get('unit'),
                    'quality_code': latest_value.get('quality_code'),
                    'procedure': observation.get('procedure')
                }
                latest_only.append(entry)

        return latest_only

    def save_latest_snapshot(self, data: Dict, filepath: str, station_names: Dict[str, str]):
        """
        Save snapshot with only the most recent single data point per parameter

        Args:
            data: Full data dictionary
            filepath: Output file path for latest snapshot
            station_names: Dictionary mapping station IDs to names
        """
        latest_only = self.create_latest_snapshot(data, station_names)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(latest_only, f, indent=2)

        print(f"Latest snapshot saved to {filepath} ({len(latest_only)} entries - one per parameter)")

    @staticmethod
    def get_last_fetch_timestamp(script_dir: str) -> Optional[datetime]:
        """
        Get the timestamp of the last successful fetch

        Args:
            script_dir: Directory where the timestamp file is stored

        Returns:
            datetime object of last fetch, or None if no previous fetch
        """
        timestamp_file = os.path.join(script_dir, BOMWaterDataFetcher.LAST_FETCH_FILE)

        if not os.path.exists(timestamp_file):
            return None

        try:
            with open(timestamp_file, 'r') as f:
                data = json.load(f)

            timestamp_str = data.get('last_fetch_timestamp')
            if timestamp_str:
                return datetime.fromisoformat(timestamp_str)
        except Exception as e:
            logging.error(f"Error reading last fetch timestamp: {e}")
            return None

        return None

    @staticmethod
    def save_last_fetch_timestamp(script_dir: str, fetch_time: datetime):
        """
        Save the timestamp of the current fetch

        Args:
            script_dir: Directory where the timestamp file should be stored
            fetch_time: datetime object of the current fetch
        """
        timestamp_file = os.path.join(script_dir, BOMWaterDataFetcher.LAST_FETCH_FILE)

        data = {
            'last_fetch_timestamp': fetch_time.isoformat(),
            'last_fetch_readable': fetch_time.strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            with open(timestamp_file, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Saved last fetch timestamp: {fetch_time.isoformat()}")
        except Exception as e:
            logging.error(f"Error saving last fetch timestamp: {e}")


def setup_logging(script_dir: str):
    """
    Configure logging for the application

    Args:
        script_dir: Directory where log files should be stored
    """
    log_dir = os.path.join(script_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'bom_fetcher_{datetime.now().strftime("%Y%m%d")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def fetch_data():
    """Fetch water data and update historical data (runs on schedule)"""

    # Get the directory where this script is located (works on any device)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Load station IDs and names from JSON file (in the same directory as this script)
        station_file = os.path.join(script_dir, 'site_list_test.json')
        logging.info(f"Loading station IDs from {station_file}...")
        station_ids = BOMWaterDataFetcher.load_station_ids(station_file)
        station_names = BOMWaterDataFetcher.load_station_info(station_file)
        logging.info(f"Found {len(station_ids)} active stations")
    except Exception as e:
        logging.error(f"Failed to load station IDs: {e}")
        return

    # Create fetcher
    fetcher = BOMWaterDataFetcher(station_ids)

    # Define directories
    historical_data_dir = os.path.join(script_dir, 'historical_data')
    summaries_dir = os.path.join(historical_data_dir, 'summaries')

    # Track overall statistics
    total_stations_processed = 0
    total_parameters_updated = 0
    total_new_measurements = 0
    fetch_start_time = time.time()

    # Record exact fetch time
    fetch_datetime = datetime.now()
    fetch_timestamp = fetch_datetime.isoformat()

    # Collect all fetched data for snapshot
    snapshot_data = {
        'fetch_timestamp': fetch_timestamp,
        'stations': {}
    }

    # Track changes for summary
    changes_summary = {
        'fetch_timestamp': fetch_timestamp,
        'stations_updated': [],
        'stations_skipped': [],
        'total_new_measurements': 0,
        'parameters_by_station': {},
        'errors': []
    }

    print("="*60)
    print("UPDATING HISTORICAL DATA")
    print("="*60)

    # Get the timestamp of the last successful fetch
    last_fetch_dt = BOMWaterDataFetcher.get_last_fetch_timestamp(script_dir)

    if last_fetch_dt:
        logging.info(f"Last fetch was at: {last_fetch_dt.isoformat()}")
        logging.info(f"Fetching new data since last fetch...")
        global_start_date = last_fetch_dt
    else:
        logging.info(f"No previous fetch found - fetching last 90 days...")
        global_start_date = None

    # Process each station
    for i, station_id in enumerate(station_ids, 1):
        logging.info(f"[{i}/{len(station_ids)}] Processing station {station_id} ({station_names.get(station_id, 'Unknown')})")

        # Use the global last fetch timestamp for all stations
        station_dir = os.path.join(historical_data_dir, station_id)
        start_date = global_start_date

        # Get available data for this station
        availability = fetcher.get_data_availability(station_id)

        if availability is None:
            error_msg = f"Failed to get data availability from BOM API"
            print(f"  ✗ {error_msg}")
            changes_summary['stations_skipped'].append(station_id)
            changes_summary['errors'].append({
                'station_id': station_id,
                'station_name': station_names.get(station_id, f"Station {station_id}"),
                'error': error_msg
            })
            continue

        if not availability:
            error_msg = f"No parameters available for this station"
            print(f"  ✗ {error_msg}")
            changes_summary['stations_skipped'].append(station_id)
            changes_summary['errors'].append({
                'station_id': station_id,
                'station_name': station_names.get(station_id, f"Station {station_id}"),
                'error': error_msg
            })
            continue

        # Recommended time series types
        recommended_procedures = [
            'DMQaQc.Merged.AsStored.1',
            'Pat4_C_B_1_PR01',
            'PR02QaQc.Merged.AsStored.1',
            'PR02AHDQaQc.Merged.AsStored.1',
            'Received.Validated.AsStored.1'
        ]

        parameter_results = []
        station_observations = []

        # Fetch observations for each available parameter
        for parameter, time_series_list in availability.items():
            # Get the best time series for this parameter
            best_ts = None

            for ts in time_series_list:
                procedure_name = ts['procedure']
                is_recommended = any(rec in procedure_name for rec in recommended_procedures)
                is_recommended = is_recommended or 'Merged' in procedure_name or 'AsStored' in procedure_name

                if is_recommended:
                    best_ts = ts
                    break

            # If no recommended found, use the first one
            if best_ts is None and time_series_list:
                best_ts = time_series_list[0]

            if best_ts:
                print(f"  Parameter: {parameter}")
                print(f"    Procedure: {best_ts['procedure']}")

                # Fetch observations
                obs_data = fetcher.get_latest_non_empty_observation(
                    station_id,
                    best_ts['parameter_href'],
                    best_ts['procedure_href'],
                    days_back=90,
                    start_date=start_date
                )

                if obs_data and obs_data['observations']:
                    for observation in obs_data['observations']:
                        # Store observation for snapshot
                        station_observations.append(observation)

                        # Merge and save parameter data to historical
                        # Use best data point only for historical updates
                        result = BOMWaterDataFetcher.merge_and_save_parameter_data(
                            station_dir,
                            station_id,
                            observation,
                            use_best_only=True  # Only use the latest/best data point
                        )

                        parameter_results.append(result)
                        print(f"    ✓ {result['status']}")

                        # Track statistics
                        new_count = 0
                        if 'added' in result['status']:
                            # Extract number from status string
                            import re
                            match = re.search(r'added (\d+)', result['status'])
                            if match:
                                new_count = int(match.group(1))
                                total_new_measurements += new_count

                        total_parameters_updated += 1

                        # Track changes for summary
                        if station_id not in changes_summary['parameters_by_station']:
                            changes_summary['parameters_by_station'][station_id] = []

                        changes_summary['parameters_by_station'][station_id].append({
                            'parameter': parameter,
                            'new_measurements': new_count,
                            'status': result['status']
                        })
                else:
                    print(f"    ✗ No new data found")

                # Small delay to avoid overwhelming the server
                time.sleep(0.5)

        # Add station data to snapshot
        if station_observations:
            snapshot_data['stations'][station_id] = {
                'station_id': station_id,
                'station_name': station_names.get(station_id, f"Station {station_id}"),
                'data_availability': availability,
                'latest_observations': station_observations
            }
            changes_summary['stations_updated'].append(station_id)

        # Create and save station summary
        if parameter_results:
            summary = BOMWaterDataFetcher.create_station_summary(station_id, parameter_results)
            BOMWaterDataFetcher.save_station_summary(summaries_dir, station_id, summary)
            total_stations_processed += 1

        # Delay between stations
        time.sleep(1)

    # Calculate total time
    total_time = time.time() - fetch_start_time
    snapshot_data['total_fetch_time_seconds'] = round(total_time, 2)

    # Complete changes summary
    changes_summary['total_new_measurements'] = total_new_measurements
    changes_summary['total_stations_updated'] = len(changes_summary['stations_updated'])
    changes_summary['total_parameters_updated'] = total_parameters_updated
    changes_summary['fetch_duration_seconds'] = round(total_time, 2)

    # Save short-term snapshots for immediate analysis
    print(f"\n{'='*60}")
    print(f"SAVING SHORT-TERM SNAPSHOTS")
    print(f"{'='*60}")

    # Use exact fetch time for filenames
    timestamp_str = fetch_datetime.strftime("%Y%m%d_%H%M%S")

    # Save the full fetch data (contains ALL measurements for all time series)
    # Note: This snapshot includes all data points, not just the best/recommended
    snapshot_file = os.path.join(script_dir, f'data_snapshot/bom_water_data_{timestamp_str}.json')
    fetcher.save_to_json(snapshot_data, snapshot_file)

    # Save simplified snapshot with only the latest single data point per parameter (for most recent status)
    # This uses only the best/recommended data point for each parameter
    latest_file = os.path.join(script_dir, f'data_snapshot_simplified/bom_water_latest_{timestamp_str}.json')
    fetcher.save_latest_snapshot(snapshot_data, latest_file, station_names)

    # Save changes summary (metadata only - uses best data points)
    # This summarizes the update process and uses only the best/recommended data points
    summary_file = os.path.join(script_dir, f'data_snapshot_summary/bom_water_summary_{timestamp_str}.json')
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(changes_summary, f, indent=2)
    print(f"Changes summary saved to {summary_file}")

    # Save the fetch timestamp for next run
    BOMWaterDataFetcher.save_last_fetch_timestamp(script_dir, fetch_datetime)

    # Print summary
    print(f"\n{'='*60}")
    print(f"UPDATE COMPLETE")
    print(f"{'='*60}")
    print(f"Stations processed: {total_stations_processed}/{len(station_ids)}")

    if changes_summary['stations_skipped']:
        print(f"Stations skipped: {len(changes_summary['stations_skipped'])}")
        for error in changes_summary['errors']:
            print(f"  - {error['station_id']} ({error['station_name']}): {error['error']}")

    print(f"Parameters updated: {total_parameters_updated}")
    print(f"New measurements added: {total_new_measurements}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    if total_stations_processed > 0:
        print(f"Average time per station: {total_time/total_stations_processed:.2f} seconds")
    print(f"\nSnapshots saved:")
    print(f"  - {snapshot_file} (FULL DATA - all measurements)")
    print(f"  - {latest_file} (SIMPLIFIED - best data point only)")
    print(f"  - {summary_file} (SUMMARY - metadata with best data points)")
    print(f"\nHistorical data updated (using best data points only):")
    print(f"  - historical_data/{{station_id}}/{{parameter}}.json")
    print(f"  - historical_data/summaries/{{station_id}}_summary.json")
    print(f"{'='*60}")

    logging.info("Fetch completed successfully")


def main():
    """Main function to run the scheduler for hourly data fetching"""

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Setup logging
    setup_logging(script_dir)

    logging.info("="*60)
    logging.info("BOM Water Data Fetcher - Hourly Scheduler Started")
    logging.info("="*60)
    logging.info("This script will fetch new data every hour")
    logging.info("Press Ctrl+C to stop")
    logging.info("="*60)

    # Run immediately on startup
    logging.info("Running initial fetch...")
    try:
        fetch_data()
    except Exception as e:
        logging.error(f"Error during initial fetch: {e}", exc_info=True)

    # Schedule to run every hour
    schedule.every(1).hours.do(fetch_data)

    logging.info("Scheduler configured - fetching data every 1 hour")

    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user")
        logging.info("="*60)


if __name__ == '__main__':
    main()
