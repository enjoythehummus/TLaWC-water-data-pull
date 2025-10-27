#!/usr/bin/env python3
"""
Bureau of Meteorology Historical Water Data Fetcher
Retrieves complete historical data from BOM's SOS2 service for trend analysis and anomaly detection
"""

import os
import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import numpy as np
from pathlib import Path


class BOMHistoricalDataFetcher:
    """Fetches complete historical water data from Bureau of Meteorology SOS2 service"""

    BASE_URL = "http://www.bom.gov.au/waterdata/services"

    # XML namespaces used in SOS2 responses
    NAMESPACES = {
        'sos': 'http://www.opengis.net/sos/2.0',
        'om': 'http://www.opengis.net/om/2.0',
        'gml': 'http://www.opengis.net/gml/3.2',
        'wml2': 'http://www.opengis.net/waterml/2.0',
        'xlink': 'http://www.w3.org/1999/xlink',
        'gda': 'http://www.opengis.net/sosgda/1.0'
    }

    def __init__(self, station_ids: List[str], output_dir: str):
        """
        Initialize the fetcher with a list of station IDs

        Args:
            station_ids: List of BOM station IDs (e.g., ['403210', '403213'])
            output_dir: Directory to store historical data files
        """
        self.station_ids = station_ids
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create summaries directory
        self.summaries_dir = os.path.join(output_dir, 'summaries')
        Path(self.summaries_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_station_ids(filepath: str) -> List[str]:
        """
        Load station IDs from a JSON file (all sites regardless of status)

        Args:
            filepath: Path to JSON file structured like site_list_test.json

        Returns:
            List of all station IDs found in the file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        station_ids = []
        for site in data.get('sites', []):
            site_id = site.get('site_id')

            # Include all sites that have a site_id
            if site_id:
                station_ids.append(str(site_id))

        return station_ids

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

    def get_historical_observations(self, station_id: str, parameter_href: str,
                                   procedure_href: str, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Get historical observations for a specific station, parameter, and time period

        Args:
            station_id: BOM station ID
            parameter_href: Full parameter URL
            procedure_href: Full procedure (time series type) URL
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary containing observation data, or None if request fails
        """
        temporal_filter = f"{start_date}/{end_date}"

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
            response = requests.get(self.BASE_URL, params=params, timeout=60)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)

            measurements = []
            metadata = {}

            for obs_data in root.findall('.//sos:observationData', self.NAMESPACES):
                obs = obs_data.find('.//om:OM_Observation', self.NAMESPACES)
                if obs is None:
                    continue

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

                metadata = {
                    'parameter': parameter_name,
                    'procedure': procedure_name,
                    'unit': unit,
                    'interpolation': interpolation
                }

                # Get all measurement points
                for point in timeseries.findall('.//wml2:point', self.NAMESPACES):
                    tvp = point.find('.//wml2:MeasurementTVP', self.NAMESPACES)
                    if tvp is None:
                        continue

                    time_elem = tvp.find('.//wml2:time', self.NAMESPACES)
                    value_elem = tvp.find('.//wml2:value', self.NAMESPACES)

                    # Get metadata (quality code, etc.)
                    point_metadata = tvp.find('.//wml2:metadata', self.NAMESPACES)
                    quality_code = None

                    if point_metadata is not None:
                        meta_data = point_metadata.find('.//wml2:DefaultTVPMeasurementMetadata', self.NAMESPACES)
                        if meta_data is not None:
                            qualifier = meta_data.find('.//wml2:qualifier', self.NAMESPACES)
                            if qualifier is not None:
                                quality_code = qualifier.get('{http://www.w3.org/1999/xlink}title', '')

                    # Check if value is not nil/empty
                    is_nil = value_elem.get('{http://www.w3.org/2001/XMLSchema-instance}nil') if value_elem is not None else True
                    value_text = value_elem.text if value_elem is not None and not is_nil else None

                    measurement = {
                        'timestamp': time_elem.text if time_elem is not None else None,
                        'value': float(value_text) if value_text is not None else None,
                        'quality_code': quality_code
                    }
                    measurements.append(measurement)

            return {
                'metadata': metadata,
                'measurements': measurements
            }

        except Exception as e:
            print(f"    Error getting observations: {e}")
            return None

    def _get_station_dir(self, station_id: str) -> str:
        """
        Get the directory path for a specific station

        Args:
            station_id: BOM station ID

        Returns:
            Directory path for the station
        """
        station_dir = os.path.join(self.output_dir, station_id)
        Path(station_dir).mkdir(parents=True, exist_ok=True)
        return station_dir

    def _load_existing_data(self, station_id: str, parameter: str) -> Optional[Dict]:
        """
        Load existing data file if it exists

        Args:
            station_id: BOM station ID
            parameter: Parameter name

        Returns:
            Existing data dictionary or None if file doesn't exist
        """
        station_dir = self._get_station_dir(station_id)
        param_filename = f"{parameter.replace(' ', '_').replace('/', '_')}.json"
        param_filepath = os.path.join(station_dir, param_filename)

        if os.path.exists(param_filepath):
            try:
                with open(param_filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"    Warning: Could not load existing file {param_filename}: {e}")
                return None
        return None

    def _get_last_timestamp(self, existing_data: Dict) -> Optional[datetime]:
        """
        Get the last timestamp from existing measurements

        Args:
            existing_data: Existing data dictionary

        Returns:
            Last timestamp as datetime object or None
        """
        measurements = existing_data.get('measurements', [])
        if not measurements:
            return None

        timestamps = [m['timestamp'] for m in measurements if m.get('timestamp')]
        if not timestamps:
            return None

        last_timestamp = max(timestamps)
        try:
            return datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
        except:
            return None

    def _create_station_summary(self, station_id: str, station_data: Dict):
        """
        Create a summary file for a specific station

        Args:
            station_id: BOM station ID
            station_data: Station data dictionary containing parameters and statistics
        """
        summary = {
            'station_id': station_id,
            'last_updated': datetime.now().isoformat(),
            'total_measurements': station_data.get('total_measurements', 0),
            'parameter_count': len(station_data.get('parameters', {})),
            'parameters': []
        }

        # Extract summary for each parameter
        for parameter, param_data in station_data.get('parameters', {}).items():
            param_summary = {
                'parameter': parameter,
                'unit': param_data.get('unit'),
                'measurement_count': param_data.get('measurement_count', 0),
                'statistics': param_data.get('statistics', {}),
                'data_range': param_data.get('data_range', {}),
                'status': param_data.get('status', 'unknown')
            }
            summary['parameters'].append(param_summary)

        # Save to summaries directory
        summary_filename = f"{station_id}_summary.json"
        summary_filepath = os.path.join(self.summaries_dir, summary_filename)

        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"   Created summary: summaries/{summary_filename}")

    def fetch_all_historical_data(self, chunk_size_days: int = 365) -> Dict:
        """
        Fetch complete historical data for all stations in manageable chunks.
        Skips data that has already been fetched. Only pulls new data.

        Args:
            chunk_size_days: Number of days to fetch in each request (default: 365)

        Returns:
            Dictionary containing summary of data fetched
        """
        start_time = time.time()

        # Recommended time series types (from Table 24 in documentation)
        recommended_procedures = [
            'DMQaQc.Merged.AsStored.1',
            'Pat4_C_B_1_PR01',  # Watercourse Discharge with flood warning data
            'PR02QaQc.Merged.AsStored.1',  # Storage Volume
            'PR02AHDQaQc.Merged.AsStored.1',  # Storage Level - AHD
        ]

        summary = {
            'fetch_timestamp': datetime.now().isoformat(),
            'stations_processed': 0,
            'total_parameters': 0,
            'total_measurements': 0,
            'new_measurements': 0,
            'skipped_parameters': 0,
            'stations': {}
        }

        for i, station_id in enumerate(self.station_ids, 1):
            elapsed = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"Processing station {i}/{len(self.station_ids)}: {station_id}")
            print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            print(f"{'='*70}")

            # Get available data for this station
            availability = self.get_data_availability(station_id)

            if availability is None:
                print(f"   No data availability information for station {station_id}")
                continue

            station_summary = {
                'station_id': station_id,
                'parameters': {},
                'total_measurements': 0,
                'new_measurements': 0
            }

            # Process each parameter
            for parameter, time_series_list in availability.items():
                print(f"\n  Parameter: {parameter}")
                print(f"  Available time series: {len(time_series_list)}")

                # Check if we already have data for this parameter
                existing_data = self._load_existing_data(station_id, parameter)

                # Select best time series (prioritize recommended/merged)
                best_ts = None
                for ts in time_series_list:
                    procedure_name = ts['procedure']
                    is_recommended = any(rec in procedure_name for rec in recommended_procedures)
                    is_recommended = is_recommended or 'Merged' in procedure_name or 'AsStored' in procedure_name

                    if is_recommended:
                        best_ts = ts
                        break

                if best_ts is None:
                    best_ts = time_series_list[0] if time_series_list else None

                if not best_ts:
                    continue

                print(f"  Selected: {best_ts['procedure']}")
                print(f"  Time range: {best_ts['begin_time']} to {best_ts['end_time']}")

                # Parse date range
                try:
                    begin_date = datetime.fromisoformat(best_ts['begin_time'].replace('Z', '+00:00'))
                    end_date = datetime.fromisoformat(best_ts['end_time'].replace('Z', '+00:00'))
                except:
                    print(f"   Could not parse date range")
                    continue

                # Determine what data we need to fetch
                all_measurements = []
                fetch_start = begin_date

                if existing_data:
                    last_timestamp = self._get_last_timestamp(existing_data)
                    if last_timestamp:
                        print(f"  Existing data found: last timestamp {last_timestamp.date()}")

                        # Check if existing data is up to date
                        if last_timestamp >= end_date:
                            print(f"  Data is up to date. Skipping.")
                            summary['skipped_parameters'] += 1

                            # Still add to summary
                            station_summary['parameters'][parameter] = {
                                'filename': f"{station_id}/{parameter.replace(' ', '_').replace('/', '_')}.json",
                                'unit': existing_data.get('metadata', {}).get('unit'),
                                'measurement_count': len(existing_data.get('measurements', [])),
                                'statistics': existing_data.get('statistics', {}),
                                'data_range': {
                                    'available_start': best_ts['begin_time'],
                                    'available_end': best_ts['end_time'],
                                    'pulled_start': existing_data.get('data_range', {}).get('start'),
                                    'pulled_end': existing_data.get('data_range', {}).get('end')
                                },
                                'status': 'skipped - up to date'
                            }
                            continue

                        # Load existing measurements and start from last timestamp + 1 day
                        all_measurements = existing_data.get('measurements', [])
                        fetch_start = last_timestamp + timedelta(days=1)
                        print(f"  Fetching new data from {fetch_start.date()} to {end_date.date()}")
                    else:
                        print(f"  Existing file has no valid timestamps. Re-fetching all data.")
                else:
                    print(f"  No existing data. Fetching complete history.")

                # Fetch data in chunks
                current_start = fetch_start
                chunk_num = 0
                new_measurements_count = 0

                while current_start < end_date:
                    chunk_num += 1
                    current_end = min(current_start + timedelta(days=chunk_size_days), end_date)

                    print(f"    Chunk {chunk_num}: {current_start.date()} to {current_end.date()}", end=" ")

                    obs_data = self.get_historical_observations(
                        station_id,
                        best_ts['parameter_href'],
                        best_ts['procedure_href'],
                        current_start.strftime('%Y-%m-%d'),
                        current_end.strftime('%Y-%m-%d')
                    )

                    if obs_data and obs_data['measurements']:
                        # Avoid duplicates by checking timestamps
                        existing_timestamps = {m['timestamp'] for m in all_measurements}
                        unique_new = [m for m in obs_data['measurements'] if m['timestamp'] not in existing_timestamps]

                        all_measurements.extend(unique_new)
                        new_measurements_count += len(unique_new)
                        print(f" {len(unique_new)} new measurements")
                    else:
                        print(" No data")

                    current_start = current_end + timedelta(days=1)
                    time.sleep(0.5)  # Rate limiting

                if all_measurements:
                    # Sort measurements by timestamp
                    all_measurements.sort(key=lambda x: x['timestamp'] or '')

                    # Calculate statistics for anomaly detection
                    values = [m['value'] for m in all_measurements if m['value'] is not None]

                    stats = {}
                    if values:
                        stats = {
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

                    # Save parameter data to file in station-specific directory
                    station_dir = self._get_station_dir(station_id)
                    param_filename = f"{parameter.replace(' ', '_').replace('/', '_')}.json"
                    param_filepath = os.path.join(station_dir, param_filename)

                    # Get metadata from existing data or new observation
                    metadata = existing_data.get('metadata', {}) if existing_data else {}
                    if obs_data and 'metadata' in obs_data:
                        metadata = obs_data['metadata']

                    parameter_data = {
                        'station_id': station_id,
                        'parameter': parameter,
                        'procedure': best_ts['procedure'],
                        'metadata': metadata,
                        'data_range': {
                            'start': best_ts['begin_time'],
                            'end': best_ts['end_time']
                        },
                        'statistics': stats,
                        'measurements': all_measurements,
                        'last_updated': datetime.now().isoformat()
                    }

                    with open(param_filepath, 'w') as f:
                        json.dump(parameter_data, f, indent=2)

                    status_msg = f"{'Updated' if existing_data else 'Created'}"
                    print(f"   {status_msg} {station_id}/{param_filename}: {len(all_measurements)} total ({new_measurements_count} new)")
                    print(f"  Statistics: mean={stats.get('mean', 0):.2f}, std={stats.get('std', 0):.2f}, min={stats.get('min', 0):.2f}, max={stats.get('max', 0):.2f}")

                    # Calculate actual pulled data range from measurements
                    pulled_start = None
                    pulled_end = None
                    if all_measurements:
                        timestamps = [m['timestamp'] for m in all_measurements if m.get('timestamp')]
                        if timestamps:
                            pulled_start = min(timestamps)
                            pulled_end = max(timestamps)

                    station_summary['parameters'][parameter] = {
                        'filename': f"{station_id}/{param_filename}",
                        'unit': metadata.get('unit'),
                        'measurement_count': len(all_measurements),
                        'new_measurements': new_measurements_count,
                        'statistics': stats,
                        'data_range': {
                            'available_start': best_ts['begin_time'],
                            'available_end': best_ts['end_time'],
                            'pulled_start': pulled_start,
                            'pulled_end': pulled_end
                        },
                        'status': 'updated' if existing_data else 'created'
                    }
                    station_summary['total_measurements'] += len(all_measurements)
                    station_summary['new_measurements'] += new_measurements_count
                    summary['total_measurements'] += len(all_measurements)
                    summary['new_measurements'] += new_measurements_count
                    summary['total_parameters'] += 1

                time.sleep(1)  # Rate limiting between parameters

            summary['stations'][station_id] = station_summary
            summary['stations_processed'] += 1

            # Create station-specific summary file
            if station_summary['parameters']:
                self._create_station_summary(station_id, station_summary)

            # Save incremental summary
            summary_file = os.path.join(self.output_dir, 'fetch_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

        total_time = time.time() - start_time
        summary['total_fetch_time_seconds'] = round(total_time, 2)

        print(f"\n{'='*70}")
        print(f"FETCH COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Stations processed: {summary['stations_processed']}")
        print(f"Total parameters: {summary['total_parameters']}")
        print(f"Parameters skipped (up to date): {summary['skipped_parameters']}")
        print(f"Total measurements: {summary['total_measurements']}")
        print(f"New measurements fetched: {summary['new_measurements']}")
        print(f"{'='*70}")

        return summary

    def update_historical_data(self, station_id: str, parameter: str) -> bool:
        """
        Update historical data for a specific station and parameter with new data

        Args:
            station_id: BOM station ID
            parameter: Parameter name

        Returns:
            True if update was successful, False otherwise
        """
        # Find existing data file in station-specific directory
        station_dir = self._get_station_dir(station_id)
        param_filename = f"{parameter.replace(' ', '_').replace('/', '_')}.json"
        param_filepath = os.path.join(station_dir, param_filename)

        if not os.path.exists(param_filepath):
            print(f"No existing data file found for {station_id} - {parameter}")
            return False

        # Load existing data
        with open(param_filepath, 'r') as f:
            existing_data = json.load(f)

        # Get last measurement timestamp
        measurements = existing_data['measurements']
        if not measurements:
            print(f"No existing measurements found")
            return False

        last_timestamp = max(m['timestamp'] for m in measurements if m['timestamp'])
        last_date = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
        current_date = datetime.now()

        print(f"Updating {parameter} for station {station_id}")
        print(f"Last data point: {last_date.date()}")
        print(f"Fetching new data from {last_date.date()} to {current_date.date()}")

        # Fetch new data
        availability = self.get_data_availability(station_id)
        if not availability or parameter not in availability:
            print(f"Parameter {parameter} not available")
            return False

        # Find matching time series
        procedure_href = existing_data.get('metadata', {}).get('procedure', '')
        time_series = None
        for ts in availability[parameter]:
            if procedure_href in ts['procedure']:
                time_series = ts
                break

        if not time_series:
            time_series = availability[parameter][0]

        # Fetch new observations
        obs_data = self.get_historical_observations(
            station_id,
            time_series['parameter_href'],
            time_series['procedure_href'],
            last_date.strftime('%Y-%m-%d'),
            current_date.strftime('%Y-%m-%d')
        )

        if not obs_data or not obs_data['measurements']:
            print(f"No new data found")
            return False

        # Merge new measurements (avoid duplicates)
        existing_timestamps = {m['timestamp'] for m in measurements}
        new_measurements = [m for m in obs_data['measurements'] if m['timestamp'] not in existing_timestamps]

        if not new_measurements:
            print(f"No new unique measurements found")
            return False

        # Add new measurements
        measurements.extend(new_measurements)
        measurements.sort(key=lambda x: x['timestamp'] or '')

        # Recalculate statistics
        values = [m['value'] for m in measurements if m['value'] is not None]
        if values:
            existing_data['statistics'] = {
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

        existing_data['measurements'] = measurements
        existing_data['last_updated'] = datetime.now().isoformat()
        existing_data['data_range']['end'] = current_date.isoformat()

        # Save updated data
        with open(param_filepath, 'w') as f:
            json.dump(existing_data, f, indent=2)

        print(f" Added {len(new_measurements)} new measurements")
        print(f" Total measurements: {len(measurements)}")

        return True


def main():
    """Main function to run the historical data fetcher"""

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load station IDs from JSON file
    station_file = os.path.join(script_dir, 'site_list_test.json')
    print(f"Loading station IDs from {station_file}...")
    station_ids = BOMHistoricalDataFetcher.load_station_ids(station_file)
    print(f"Found {len(station_ids)} stations")

    # Set output directory
    output_dir = os.path.join(script_dir, 'historical_data')

    # Create fetcher
    fetcher = BOMHistoricalDataFetcher(station_ids, output_dir)

    # Fetch all historical data
    print("\nFetching complete historical water data from BOM SOS2 service...")
    print("(This may take considerable time depending on the number of stations and data volume)\n")

    summary = fetcher.fetch_all_historical_data(chunk_size_days=365)

    # Save final summary
    summary_file = os.path.join(output_dir, 'fetch_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_file}")
    print(f"Data files saved to {output_dir}/")


if __name__ == '__main__':
    main()
