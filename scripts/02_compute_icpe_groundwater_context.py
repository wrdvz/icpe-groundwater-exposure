"""Compute local groundwater context around ICPE sites.

Planned method:
- project ICPE sites and groundwater stations to Lambert-93
- find groundwater stations within 20 km of each ICPE site
- compute median and mean 20-year groundwater variation
- flag the signal as solid when at least 5 stations are available
"""


def main() -> None:
    raise NotImplementedError("Implement the 20 km ICPE groundwater context indicator.")


if __name__ == "__main__":
    main()
