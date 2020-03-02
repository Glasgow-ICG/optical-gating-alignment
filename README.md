# Reference period alignment algorithms for adaptive prospective optical gating for time-lapse 3D fluorescence microscopy

[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/Glasgow-ICG/optical-gating-alignment/?ref=repository-badge)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Chas Nelson and Jonathan Taylor

### School of Physics and Astronomy, University of Glasgow, UK

Cardiac diseases account for more deaths worldwide than any other cause.
The zebrafish is a commonly used and powerful model organism for investigating cardiac conditions with a strong connection to human disease.
This is important for furthering biomedical sciences such as developing new disease models or drugs to combat those diseases. 

Prospective optical gating technologies allow phase-locked, 3D, time-lapse microscopy of the living, beating zebrafish heart without the use of pharmaceuticals or electrical/optical pacing [1].
Further, prospective optical gating reduces the data deluge and processing time compared to other gating-based techniques.

This repository contains the algorithms that allow long-term phase-lock to be maintain over hours and days by aligning new reference periods to historical periods.

1. Taylor, J.M., Nelson, C.J., Bruton, F.A. et al. Adaptive prospective optical gating enables day-long 3D time-lapse imaging of the beating embryonic zebrafish heart. Nat Commun 10, 5173 (2019) doi:[10.1038/s41467-019-13112-6](https://dx.doi.org/10.1038/s41467-019-13112-6)

## Other Repositories That Use This Code

1. The data repository for [1] has a copy of the cross-correlation method codes contained in doi:[10.1038/s41467-019-13112-6](http://dx.doi.org/10.1038/s41467-019-13112-6).
2. Our Raspberry Pi-based open optical gating solution uses this repository as a submodule here:[GlasgowICG/open-optical-gating](https://github.com/Glasgow-ICG/open-optical-gating).
