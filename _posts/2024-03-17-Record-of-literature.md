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




###### **冲突与交互**（https://www.beyondintractability.org/artsum/folger-conflict）
冲突可能是破坏性的也可能是建设性的，他们一般性的描述了冲突的本质，已经冲突建设性和破坏性的特征。所有的冲突都以某种程度的紧张、不确定性和不愉快为特征。冲突局势通常是脆弱的，看似微不足道的事件可能会对冲突的走向产生深远的影响。此外，结局糟糕的冲突望望比结局好的冲突更令人难忘。由于这些因素，人们往往对冲突持有消极的态度，并试图压制或避免冲突。 

"冲突是相互依存的人之间的互动，他们认为目标不相容，并且在实现这些目标过程中相互干扰"，定义强调冲突的互动性质，冲突互动的范围从公开的对抗和竞争到试图压制和避免对抗。冲突几乎在任何社会情境中都可能发生，其重要性从微不足道到深刻不等。 冲突的中心是对不相容的目标和干扰的看法，即使这种看法实际上并不准确，冲突也会发生。此外，即使各方对自己的目标缺乏清晰的了解，冲突也可能发生。虽然，有些冲突可能仅仅源于误解或者沟通不顺，但大多数冲突都是由于实际的不兼容造成的。
冲突各方的相互依存的，每一方的行为都会影响另一方产生影响，或者帮助或者阻碍对方的利益。在冲突局势中，这种相互依存通过竞争或合作的混合激励表现出来。各方对彼此动机的看法会影响激励的平衡，竞争或者合作动机的平衡对于确定冲突互动的方向非常重要。（在交通冲突中能否引入类似这样的分析行为，冲突中两个车都减速，单一方面减速，避让，三种模式：机动车和非机动车同时避让行为，机动车避让非机动车，非机动车避让机动车，）

生产性和破坏性冲突，现实冲突和不现实冲突的社会学区别。现实冲突集中子啊实质性问题上的分歧，目标是解决分歧，参与这通常利用多种技术解决次类冲突。这种处理冲突的灵活性是富有成效的冲突的一个特征。相反，非现实冲突的目标是击败或者摧毁对手，参与者通常使用武力、侵略和胁迫。他们的做法僵化，二这种僵化望望会导致冲突升级。缺乏灵活性是破坏性冲突的标志。 富有成效的冲突，寻求一种能够让所有相关方都满意的解决方案。他们往往以双赢为导向，一般说来，各方愿意努力解决分歧，直到找到双方都满意的解决方案。破会性冲突尝尝需求彼此的失败。他们倾向于以输赢为导向，作者警告说，使用投票来解决冲突可能引发输赢心态，因此可能助长破坏性冲突。

这两种形式的冲突都可能涉及对强势地位的激烈竞争。当立场两极分化并且各方都固守自己的立场时，这种竞争就会变得具有破坏性。
冲突理解为互动，冲突并不完全由任何一方控制，冲突应该从行为周期的角度来看待，行为周期具有独立于个人行为总和的特征。这种行为模式望望会形成自我强化的循环例如升级模式。

各方的行为既是如此，也是预测性的，参与者对于对方的最后一步做出反应，并期待他们的下一步。这种预测因素涉及解释对方的动机，并可能使理解冲突各方的想法变得相当困难，它还可以产生无限的螺旋，因为我试图预测你会预测什么，我会预测你的预测关于我的预测。
