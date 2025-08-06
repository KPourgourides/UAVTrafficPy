---
title: 'UAVTrafficPy: Package for analysis & visualization of UAV-based traffic data in Python'
tags:
  - Python
  - data science
  - traffic analysis
  - visualization
authors:
  - name: Konstantinos Pourgourides
    orcid: 0009-0009-7526-8195
    affiliation: 1
  - name: Yiolanda Englezou
    orcid: 0000-0003-2826-0501
    affiliation: 1
  - name: Christos Panayiotou
    orcid: 0000-0002-6476-9025
    affiliation: 1
  - name: Stelios Timotheou
    orcid: 0000-0002-3617-7962
    affiliation: 1
affiliations:
 - name: KIOS Research and Innovation Center of Excellence, and the Department of Electrical and Computer Engineering, University of Cyprus
   index: 1
date: 6 August 2025
bibliography: paper.bib
---

# Summary

This paper presents UAVTrafficPy, an open-source Python package for scientific analysis and visualization of UAV-based traffic data. This tool enables users to extract important information and make insightful visualizations regarding UAV-based traffic datasets, with a primary focus on urban environments, such as arterial roads and signalized intersections. UAVTrafficPy was developed within the ERC-funded project [URANUS](https://uranus.ucy.ac.cy/) to support researchers and students in UAV-driven transportation research, by offering a ready-to-use framework for data analysis and visualization.

# Statement of need

The introduction of UAV-based data collection methods in transportation research [@BARMPOUNAKIS202050],[@highDdataset] has unlocked new possibilities which were inaccessible in the past due to the limitations of traditional sensor-based methods, such as loop detectors beneath the road network, static cameras and probe vehicles equiped with tracking technologies. These conventional methods suffer from different drawbacks, such as sparse data inflow, limited coverage of the traffic network, high installation and maintenance costs, and the inability to capture and store high quality spatio-temporal information for individual vehicles. Usually, they have to be combined together, in order to compensate for their inadequencies and deliver reliable results [@7676331].

On the other hand, the rapid deployment of coordinated and camera-equipped UAVs above traffic networks enables researchers to overcome these limitations, as they can acquire large volumes of data for all individual vehicles in a potentially large area, with high spatio-temporal resolution. This can transform the landscape of data science in transportation, provided that appropriate tools are developed. 

[UAVTrafficPy](https://github.com/KPourgourides/UAVTrafficPy) is an initial attempt to bridge the gap between UAV-based data collection and meaningful traffic analysis. It provides users with tools for reconstructing and visualizing vehicle trajectories in various forms, calculating time-dependent speed and acceleration profiles, and computing the cumulative distance travelled by vehicles, leading to insightful space-time diagrams. Additionally, it provides tools for identifying network characteristics, such as the number and spatial distribution of lanes, and calculating important variables for the calibration of car-following models, such as relative dynamic gaps and speed differences between vehicles and their leaders. Finally, it enables the extraction of useful information regarding signalized intersections, such as the duration of traffic light phases, and the physical length and dissipation time of queues.

# Software Functionality

UAVTrafficPy was developed and tested in Python 3.12, and acts as a standalone package for analysis and visualization of UAV-based traffic data. It is not an extension of any already existing software. Since UAV traffic datasets often come in compact, non-intuitive formats that vary significantly across sources, UAVTrafficPy introduces a universal input structure based on Python dictionaries. The tool requires only minimal information to operate, such as vehicle IDs, vehicle types (e.g., car, motorcycle etc.), and 2D positional coordinates (longitude and latitude in the WGS84 system) labeled by time for every vehicle. 

Once a user provides this information in the appropriate format, which is thoroughly explained in the [walkthrough](https://github.com/KPourgourides/UAVTrafficPy?tab=readme-ov-file#acquiring-the-data-in-the-correct-format) on the repository's frontpage, UAVTrafficPy can perform a wide range of tasks. Examples include computing time-dependent speed and acceleration profiles, and reconstructing vehicle trajectories at signalized intersections based on the existent routes, which are depicted in \autoref{fig:1} and \autoref{fig:2} respectively. These tasks were executed by using the file `20181024_d2_0900_0930.csv` from the open-source pNEUMA dataset.

![Reconstruction of UAV-based vehicle trajectories based on their routes in a signalized intersection. \label{fig:1}](images/trajectories.png)

![Speed and acceleration of a random vehicle as a function of time using its UAV-based trajectory. \label{fig:2}](images/speed_acceleration.png)

# Acknowledgements

This work is supported by the European Union (i. ERC, URANUS, No. 101088124 and, ii. Horizon 2020 Teaming, KIOS CoE, No. 739551), and the Government of the Republic of Cyprus through the Deputy Ministry of Research, Innovation, and Digital Strategy. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. 

# References
























