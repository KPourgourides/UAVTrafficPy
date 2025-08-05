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
affiliations:
 - name: KIOS Research and Innovation Center of Excellence, University of Cyprus, Nicosia, Cyprus
   index: 1
   ror: 037ez2r19
date: 4 August 2025
bibliography: paper.bib
---

# Summary

This paper introduces UAVTrafficPy, an open-source Python package for scientific analysis and visualization of UAV-based traffic data. This tool provides users with the ability to extract important information and make insightful visualizations regarding UAV-based traffic datasets, with a significant focus on signalized intersections in urban networks. UAVTrafficPy was developed within the ERC project "URANUS" to help researchers or students working in UAV-related transportation research with their tasks by providing ready-to-use tools for analysis and visualization.

# Statement of need

The recent introduction of UAV-based data collection methods in transportation research has unlocked new of possibilities which were inaccessible in the past due to the limitations of traditional sensor-based methods. These conventional data collection methods include loop detectors beneath the road network, static cameras, as well as GPS or bluetooth signals from drivers' mobile devices. Despite their advantages, all of them suffer from different drawbacks, such as sparse data inflow, limited coverage of the traffic network, and the inability to capture and store high quality spatio-temporal information for individual vehicles, amongst others.

On the other hand, the rapid deployment of coordinated and camera-equipped UAVs above traffic networks allows researchers to overcome these limitations, as they have the ability to acquire large volumes of data for all individual vehicles in a potentially large area, with high spatio-temporal resolution. This can transform the landscape of data science in transportation, provided that new and appropriate tools are developed. 

[UAVTrafficPy](https://github.com/KPourgourides/UAVTrafficPy) is an initial attempt to bridge the gap between UAV-based data collection and meaningful traffic analysis. It provides users with tools for extracting and visualizing vehicle trajectories in various forms, calculating position-derivative quantities such as speeds and accelerations, and computing the cumulative distance travelled by vehicles as a function of time, which leads to insightful space-time diagrams. Additionally, it provides tools for identifying network characteristics such as the number and spatial distribution of lanes, calculating quantities that are useful for the calibration of car-following models such as relative dynamic gaps and speed differences. Finally, it enables the extraction of useful information regarding signalized intersections, such as the duration of traffic light phases, and the length and dissipation time of queues.

# Software Functionality

UAVTrafficPy was developed in Python, and acts as a standalone package for analysis and visualization of UAV-based traffic data. It is not an extension of any already existing software. Since UAV traffic datasets often come in compact, non-intuitive formats that vary significantly across sources, UAVTrafficPy introduces a universal input structure based on Python dictionaries. The tool requires only minimal information to operate, such as vehicle IDs, vehicle types (e.g., car, motorcycle etc.), and 2D positional coordinates (longitude and latitude in the WGS84 system) labeled by time for every vehicle. 

Once a user provides this information in the appropriate format, which is thoroughly explained in the [walkthrough](https://github.com/KPourgourides/UAVTrafficPy?tab=readme-ov-file#acquiring-the-data-in-the-correct-format) on the repository's frontpage, UAVTrafficPy can perform a wide range of tasks. Examples include computing time-dependent speed and acceleration profiles, and extracting vehicle trajectories at signalized intersections based on the existent routes, which are depicted in \autoref{fig:1} and \autoref{fig:2} respectively. These tasks were executed by using the file `20181024_d2_0900_0930.csv` from the open-source pNEUMA dataset.

![UAV-based vehicle trajectories separated based on their routes in a signalized intersection. \label{fig:1}](images/trajectories.png)

![Speed and acceleration of a random vehicle as a function of time using its UAV-based trajectory. \label{fig:2}](images/speed_acceleration.png)

# Acknowledgements

I would like to thank KIOS Research & Innovation Center of Excellence for funding my work through the URANUS project, which received funding from the European Research Council (ERC) under the ERC Consolidator Grant scheme (Grant agreement No. 101088124).

# References













