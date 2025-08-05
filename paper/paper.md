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

The recent introduction of UAV-based data collection methods in transportation research has unlocked new of possibilities which were inaccessible in the past due to the limitations of traditional sensor-based methods. These conventional methods include static loop detectors beneath the road network, static cameras, as well as floating car data collected from GPS or bluetooth signals from drivers' mobile devices, often suffer from drawbacks, such as sparse data, limited coverage of the traffic network, and the inability to capture and store high quality spatio-temporal information for individual vehicles. 

On the other hand, with the quick deployment of coordinated, camera-equipped drones above a traffic network, researchers have the ability to acquire large volumes of traffic data for all individual vehicles in the network, with high spatio-temporal resolution. This has the potential to transform the landscape of data science in transportation research, as long as new and appropriate tools are developed. 

[UAVTrafficPy](https://github.com/KPourgourides/UAVTrafficPy) is an initial attempt to bridge the gap between UAV-based data collection and meaningful data analysis, as it provides users with tools to extract and visualize vehicle trajectories in various useful forms, calculate position-derivative quantities such as speeds and accelerations, calculate the cumulative distance travelled by vehicles as a function of time, identify network characteristics such as the number and spatial boundaries of lanes, calculate quantities that are useful for the calibration of car-following models, such as relative dynamic gaps and speed differences, and finally extract useful information regarding intersections, such as the duration of traffic light phases, and the length and dissipation time of queues.

# Software Functionality

UAVTrafficPy was developed in Python3, and acts as a standalone package for analysis and visualization of UAV-based traffic data; it is not an extension of any already existing software. Different UAV-based traffic datasets use different formats, that are oftentimes very compact and thuis not intuitive, so UAVTrafficPy was designed to take the input data in one universal format, based on Python dictionaries. The tool only needs minimal information to work, such as vehicle IDs, vehicle types (e.g., car, motorcycle etc.), and 2d position coordinates (based on the WGS84 system, i.e. longitiudinal and latitudinal coordinates) labeled by time for every vehicle. Once a user provides this information in the appropriate format (which is thoroughly discussed in the walkthrough in the main page of the repository), UAVTrafficPy can execute a large number of tasks, which are mentioned in the [statement of need](#statement-of-need) section. Some of these tasks, such as speed and acceleration extraction for a vehicle as a function of time, and trajectory extraction based on vehicle routes in a signalized intersection, are depicted in \autoref{fig:1} and \autoref{fig:2} respectively.

![UAV-based vehicle trajectories separated based on their routes in a signalized intersection. \label{fig:1}](images/trajectories.png)

![Speed and acceleration of a random vehicle as a function of time using its UAV-based trajectory. \label{fig:2}](images/speed_acceleration.png)

These tasks were executed by using the file `20181024_d2_0900_0930.csv` from the open-source pNEUMA dataset.

# Acknowledgements

I would like to thank KIOS Research & Innovation Center of Excellence for funding my work through the URANUS project, which received funding from the European Research Council (ERC) under the ERC Consolidator Grant scheme (Grant agreement No. 101088124).

# References












