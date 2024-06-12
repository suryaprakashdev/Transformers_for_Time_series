# GeoXts



## Description

GeoXts is a library for benchmarking time series classification. It is more focused on Geological usecase of trying to understand the sub-surface soil formations around the oil wells. 
The formation tops are refferred to as Markers. At present the marker depths are calculsted by expert geologist. This library is manly focussed on the automization of this process, for the new wells drilled.

We have an implementation of GradCAM focussed on time series data, which helps to understand the feature importance in multivariate time series. 


## Components of GeoXts

![Block diagram of the key components of GeoX](/Images/Block_diagram.png) 


## Data files

The datasets of well logs and the marker depths have been extracted from the state government websites of USA. The datset for each well is extrated seperately and then joined together to create the whole dataset.
    
A sample dataset is provided, just to understant the format of the final created dataset. 

- [ ] [Colorado dataset](https://ecmc.state.co.us/data.html#/cogis) 
- [ ] [Wyoming dataset](https://pipeline.wyo.gov/wellchoiceMenu2.cfm?oops=ID88107&Skip=Y) 
- [ ] [UEA](https://www.timeseriesclassification.com/) is a benchmark dataset for multivariate time series classification. It can be downloaded from the website or using the tsai library.



## Test and Deploy

We have tutorial notebooks on the deployment of the functions for different use cases.

- [ ] [Marker Propagation](https://gitlab.lisn.upsaclay.fr/salimath/geox/-/blob/main/Marker_propagation.ipynb?ref_type=heads)
- [ ] [Marker Propagation benchmark](https://gitlab.lisn.upsaclay.fr/salimath/geox/-/blob/main/Benchmark_Colorado.ipynb?ref_type=heads)
- [ ] [UCR Tutorial](https://gitlab.lisn.upsaclay.fr/salimath/geox/-/blob/main/UCR_tutorial.ipynb?ref_type=heads)
- [ ] [Gradcam tutorial](https://gitlab.lisn.upsaclay.fr/salimath/geox/-/blob/main/Gradcam_tutorial.ipynb?ref_type=heads)

***

The geoxts folder will be converted into a libraray.
## Installation

To install all the packages needed for to run the file, please execute the Install.ipynb.


## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
Attribution-NonCommercial-ShareAlike 4.0 International

