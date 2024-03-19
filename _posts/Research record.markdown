The challenge in detection and camera calibration.

![image-20240313114730807](E:\Github\yanzhangcun.github.io\_posts\image-20240313114730807.png)



















Research processes record:

(3-14-2024)

1. **Clean the interaction event dataset**: Extract real interaction event, confirmed by manuals  check with ground truth video, from the orginal datasets which were detected by the risk index with Time to crash(TTC) and Anticipated collision time(ACT).

   1. Note, the dataset should contain the variable related to entropy.

      *r'D:\dataset\Intersection\Data_processing\model\data\TTC\Data_use/model_data.csv'*

      *r'D:\dataset\Intersection\Data_processing\model\data\ACT\use_data/ACT_variable_data.csv'*

   2. 

2. **Add some event from the ground truth video about the process of interaction.**

   

3. **Redefine the variables related to crash risk in interaction.**  

   1. For the TTC dataset, calculated by the GCM_example_TTC.ipynb

      1. first, delete the variable without effect on crash risk in interaction.

      ​	

   2. For the ACT dataset, processing by the GCM_example_ACT.ipynb

   

4. **Modeling the crash risk for interaction of MVs and NMVs with Linear regression analysis for risk model.**

   

5. **Modeling the crash risk for interaction of MVs and NMVs by Causal inference model **

   

6. **Discussing the insight of the result of Model**



**There some questions should be clearly understood:**

General definition: A traffic conflict is an event involving two or more road users, in which the action of one user causes the other user to make an evasive maneuver to avoid a collision.



