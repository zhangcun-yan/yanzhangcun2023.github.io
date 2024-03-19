---
layout:     post
title:      The Material interaction behaviour research
subtitle:   literature about interaction behaviour
date:       2024-03-16
author:     Zhangcun Yan
header-img: img/post-bg-recitewords.jpg
catalog:    false
tags:
    - 资料总结
---


***Record of literature.***

###### **Title:** *How do cyclists interact with motorized vehicles at unsignalized intersections? Modeling cyclists’ yielding behavior using naturalistic data* ,(https://www.sciencedirect.com/science/article/pii/S0001457523002038#:~:text=Abstract,traffic%20scenarios%20has%20been%20decreasing.) ######

For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.

**Statement:** 

1) we search for the objects labeled “cyclists” and “vehicles” in the trajectory dataset and only used event which included a single car and a single bicycle approaching the intersection (and no other road user present). An interaction event can be defined as occurring when two road users share the road., they may try to communicate with each other and probe the other’s intent to fellow a safe and comfortable path.
2) we extracted the kinematic information for the involved road users from the trajectory dataset and annotated the correspondent videos to record additional information. Defined all the variables extracted for each interaction event(one hand: contains the variables extracted from the trajectory dataset, the second part describes the variables acquired from the video annotation)
3) Three variables indicate implicit communication between the cyclist and motorized vehicle: whether they were pedaling, looking toward the approaching vehicle, and making any hand gestures(as a sign to let the vehicles cross first or to thank the driver). these variables as time series, meaning that the variables were continuously annotated.
4) The PET and *projected PET*(predicted value, with the constant value of speed) were calculated and the DTA was recorded. The decision to yield was recorded as a binary variable: 0 if the car driver yielded and 1 if the cyclist yielded.
5) 8m was used as the decision point for the cyclists and sought to determine the occurrence of cyclists’ visual cues(pedaling, looking at the motorized vehicle, and hand gesture) before this point. the model also aims to predict the decision to yield(based on these cues) before the cyclist arrives at the decision point.
6) we extracted the cyclist’s and vehicle’s speeds from the first moment that they were visible to each other(19 and 15 from the intersection point for the bike and vehicle, respectively).

**Detail of method** 

​	*The new index:* **The difference in time to arrival(DTA)**  to assess which road user arrived sooner at the intersection and by how much.

​	*The variable related to the event model:* **dependent variable**: the interactions were assigned a severity level from 0(low severity) to 4 (crash).  **Independent variables**: the age categories: adults, elderly and children; weather condition, lighting conditions: dry or night, wearing helmet or not, the type of bike. 

**Problem in this paper:** 

The PET, interaction severity level, and projected PET were also coded each event. However, we did not include them in the model because they cannot be used to predict the yielding decision. These variables are processed once the yielding decision has already been made.

The novel of this paper is that none of them involved cyclists’ visual cues in their predictive models.

**The limitation of this paper:**  they only considered interactions between motorized vehicles and cyclists when no other road user was present, in order to avoid other influencing factors and have a clean environment.

###### **Title:** predicting conflict risk for vehicle queues at congested intersections using unsupervised clustering and LSTM models.   [TRC_draft](E:\Academic\project\Drivingbehaviormoding\literature review/TRC-23-02318.pdf)

This paper focuses on the problem of  “the conflict occurring in a VQ may cause a chain reaction resulting in sever collisions. ” This paper proposes a framework for assessing and predicting conflict risks for VQ using high-resolution trajectory data. 

The objective of this paper is to predict the conflict for VQ with a three steps multiagent-based method which contain path band, multiagent-based approach, multi-dimensional feature sequence( describe each VQ’s motion,  developed traffic analytics are used to label conflict risk, the LSTM model are developed to predict conflict risks for VQ).

**Good idea:**  existing studies were estimated at the individual level. specifically, each traffic participant within the intersection region is viewed as an object. Any disturbance of VQ’s movement may cause a chain reaction that leads to a sever collision. 

The creation of conflict areas was used primarily for post-analyses of interactions between conflicting agents at intersections. However, traffic conditions in the real life are very complex and the occurrence of angle conflicts may not be confined to a specific area.

*The summary of recently research workflow of establishing different types of conflict risk prediction models is similar.* **First,** high-resolution trajectory or motion data of traffic participants were collected on sites. **Second,** traffic parameters and conflict risk were extracted and measured from the collected data, respectively. **Third.** prediction models were built using the traffic parameters and conflict risk(labels) as input and output, respectively. The main difference lie in the selection of traffic variables, the consideration of conflict types, and the use of specific models. 

The shortcoming of existing studies is that only a single type of conflict was considered when establishing prediction models.

**The main framework of this paper:** 

**Step 1**: *grid map-based clustering*  a path band is first creat for each vehicle based on motion data. then, all path bands are rasterized into a grid map. **The connectivity among vehicles is estim**  



###### **Title:** Modeling lateral interferences between motor vehicles and non-motor vehicles: A survival analysis based approach,(https://www.sciencedirect.com/science/article/pii/S1369847815001096).

**background:** mixed traffic flow consisting for motor vehicles and non-motor vehicles is a typical inhomogeneous traffic flow. The interactions between motor vehicles and non-motor vehicles extert significant influence on traffic performance and safety. Even a low proportion of non-motor vehicles tend to significantly reduce the operating speed of traffic flow.
While traveling under mixed traffic circumstance, both motor vehicles drivers and non-motor vehicles drivers are prone to adjuesting speed or position to avoidance interference. such avoidance behavior is important for drivers to keep safe. 

**The lateral interference leads to the fellowing situations:** (a) The motor vehicles move straightly at uniform speed, with the non-motor vehicles decelerating or moving away from motor vehicles. (b) The non-motor vehicles move straightly at uniform speed, with the motor vehicles decelerating or changing direction. (c) Both motor vehicles and non-motor vehicles decelerate or change direction to avoid the lateral interference.

**useful of this paper:** this paper could be used in the literature review about the interaction behaviour between motor vehicles and non motor vehicles.


###### **Title:** Identification of evasive manoeuvres in traffic interactions and conflicts.


**Title:** Evaluating pedestrian vehicle interaction dynamics at un-signalized intersections: A proactive approach for safety analysis,(https://www.sciencedirect.com/science/article/pii/S0001457519303847)


**Title:** Relevance to human factors/Relevance to ergonomics theory,(https://www.tandfonline.com/doi/full/10.1080/1463922X.2020.1736686)

**Title:** Simulation of Signalized Intersection with Non-Lane-Based Heterogeneous Traffic Conditions Using Cellular Automata, https://journals.sagepub.com/doi/pdf/10.1177/03611981231211317

**Title:** Drivers’ skills and behavior vs. traffic at intersections, (https://arxiv.org/pdf/2101.03351.pdf)

**Title:** zhaojing


###### **Title:**  (https://www.jmir.org/2022/8/e36314/PDF)


##### **Learning Human Driving Behaviors with Sequential Causal Imitation Learning**  (https://www.google.com.hk/search?q=learning+human+driving+behaviors+with+sequential+causal+imitation+learning&rlz=1C1CHZN_enCA1045CA1045&oq=Learning+Human+Driving+Behaviors+with+Sequential+Causal+Imitation+Learning&gs_lcrp=EgZjaHJvbWUqBwgAEAAYgAQyBwgAEAAYgATSAQcxODRqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8#ip=1)


##### **Fuzzy Causal Inference Model for Human-Vehicle Cooperative Decision-Making Modeling**(https://dl.acm.org/doi/pdf/10.1145/3629606.3629639)