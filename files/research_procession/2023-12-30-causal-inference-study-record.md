---
layout:     post
title:      Causal inference and merchin learning
subtitle:   Notebook about causal inference
date:       2023-12-30
author:     Zhangcun Yan
header-img: img/home-bg-art.jpg
catalog:      true
tags:
    - technology notebook
---


How to upload your project to Github！

虽然接触因果推断这个理论快一年多了，自己没有系统的梳理过，虽然也看了不少方面的书籍，总觉的过于零碎，没有系统的梳理和总结一下。这里借助论文需要，浏览网页的知识分享，这边对因果推断和因果发现的理论进行梳理和总结。

借鉴资料<a href="https://zhuanlan.zhihu.com/p/403098221">causal inference summary </a>>



**Some video about causality inference:** (人大高瓴视频：https://gsai.ruc.edu.cn/addons/video/video/play.html?id=18)



**orthogonal / double machine learning**

双机器学习时一种估计（异质性）治疗效应的方法，当所有潜在的混杂或者对照，同时对数据采集过程中治疗决策和观测结果产生影响，也可能是多维度的。存在多个混杂因素，经典的统计方法是否适用？他们对于治疗和结果的影响不能通过参数函数（非参数）得到令人满意的建模。后两个问题都可以通过机器学习技术来解决。

该方法将问题简化为首先估计两个预测任务：1.预测控制结果；2.从对照中预测治疗。

该方法在最后阶段的估计中结合这两个预测模型，从而创建异质治疗效果的模型。该方法允许将任意机器学习算法用于两个预测任务。同时保持与最终模型相关的许多有利的统计特性（小均方误差，渐近正态性、置信区间的构造）。

**WHY WE NEED CAUSAFL INFERENCE?**  reason as why is something happening? are sales increasing because of coupon code? causal inference id the godfather of actionable prescriptions. It’s not something new. economists have been using it for years to answer questions liks. Does immigration cause unemployment?



