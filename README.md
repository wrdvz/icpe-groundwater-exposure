# ICPE groundwater exposure in France

This project extends the 20-year groundwater trend signal produced in
[`groundwater-france-trends`](../groundwater-france-trends) and crosses it with
French ICPE industrial sites.

The goal is to make the hydrological signal more actionable: identify industrial
facilities located in areas where nearby groundwater monitoring stations show a
long-term declining, stable, or rising trend.

## Analytical idea

For each ICPE site, the project will compute a local groundwater trend signal
from monitoring stations located within 20 km of the site.

Planned indicators:

- local median groundwater variation over 2005-2025, in cm
- local mean groundwater variation over 2005-2025, in cm
- number of stations within 20 km
- support flag: solid signal when at least 5 stations are available
- visual marker for declining, stable, or rising groundwater context

This indicator should be read as a contextual hydrological exposure signal, not
as a complete operational risk score for each facility.

## Relationship to the groundwater trend project

The first project establishes the national hydrological signal:

- station-level groundwater trends
- BDLISA outcropping aquifer aggregation
- national map of 20-year groundwater evolution

This second project uses that signal as an input layer and adds ICPE sites as an
operational lens.

## Planned pipeline

1. Prepare groundwater trend inputs from `groundwater-france-trends`.
2. Fetch or load ICPE sites and classify them by sector.
3. Compute a 20 km local groundwater context around each ICPE site.
4. Generate an interactive map with ICPE markers and groundwater background.
5. Summarize exposure patterns by sector, region, and signal robustness.

## Current repository state

This repository has been initialized from the first groundwater trend project,
with the heavy raw database and BDLISA source files excluded. It contains the
processed groundwater trend outputs needed as a starting point for the ICPE
analysis.

## Data sources

Groundwater trend inputs are derived from:

- ADES / Hubeau groundwater level observations
- BDLISA hydrogeological entities
- `groundwater-france-trends` processing outputs

ICPE source data still needs to be selected and documented.

## Author

Edward Vizard  
Parallaxe processing
