The challenge in detection and camera calibration.

![image-20240313114730807](E:\Github\yanzhangcun.github.io\_posts\image-20240313114730807.png)

##### the paper about how to extract the trajectory from video.(https://iopscience.iop.org/article/10.1088/1742-6596/1903/1/012040/pdf)



https://www.e3s-conferences.org/articles/e3sconf/pdf/2021/29/e3sconf_eem2021_01040.pdf



Define the variable of interaction 



![Figure 3](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-021-82331-z/MediaObjects/41598_2021_82331_Fig3_HTML.png)



![图6](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-021-82331-z/MediaObjects/41598_2021_82331_Fig6_HTML.png)





![图5](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-021-82331-z/MediaObjects/41598_2021_82331_Fig5_HTML.png)

![图7](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-021-82331-z/MediaObjects/41598_2021_82331_Fig7_HTML.png)







![An external file that holds a picture, illustration, etc. Object name is jmir_v24i8e36314_fig1.jpg](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9472037/bin/jmir_v24i8e36314_fig1.jpg)





Research processes record:

(3-14-2024)

1. **Clean the interaction event dataset**: Extract real interaction event, confirmed by manuals  check with ground truth video, from the orginal datasets which were detected by the risk index with Time to crash(TTC) and Anticipated collision time(ACT).

   1. Note, the dataset should contain the variable related to entropy.

      *r'D:\dataset\Intersection\Data_processing\model\data\TTC\Data_use/model_data.csv'*

      *r'D:\dataset\Intersection\Data_processing\model\data\ACT\use_data/ACT_variable_data.csv'*

   2. 

2. **Add some event from the ground truth video about the process of interaction.**






1. **Redefine the variables related to crash risk in interaction.**  

   1. For the TTC dataset, calculated by the GCM_example_TTC.ipynb

      1. first, delete the variable without effect on crash risk in interaction.

      ​	

   2. For the ACT dataset, processing by the GCM_example_ACT.ipynb

   

2. **Modeling the crash risk for interaction of MVs and NMVs with Linear regression analysis for risk model.**

   

3. **Modeling the crash risk for interaction of MVs and NMVs by Causal inference model **

   

4. **Discussing the insight of the result of Model**



**There some questions should be clearly understood:**

General definition: A traffic conflict is an event involving two or more road users, in which the action of one user causes the other user to make an evasive maneuver to avoid a collision.



