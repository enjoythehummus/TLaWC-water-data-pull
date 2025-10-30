# TLaWC-water-data-pull

This program pulls the most recent water data from the Australian Bureau of Meteorology
through their API using KVM requests.

Input: site_list_test.json
    - this input file is used to make requests to the API using the "site_id" as a key
        to request all recent data since the last time a request was made