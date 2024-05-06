---
layout:     post
title:      Git
subtitle:   upload the file from local to github
date:       2023-12-18
author:     Zhangcun Yan
header-img: img/post-bg-universe.jpg
catalog:      true
tags:
    - technology notebook
---

###  Git
How to upload your project to Github！

There some steps :

```git
git clone 
git status
git add .
git commit -m "my changes" 
git remote add origin https://github.com/zinmyoswe/React-and-Django-Ecommerce.git
git push -u origin master
```

summary-errors common:

[Pushing to Git returning Error Code 403 fatal: HTTP request failed](https://stackoverflow.com/questions/7438313/pushing-to-git-returning-error-code-403-fatal-http-request-failed)

So you need to change your repo config on your PC to ssh way:

1. Edit `.git/config` file under your repo directory.
2. Find `url=`entry under section `[remote "origin"]`.
3. Change it from:
   `url=https://MichaelDrogalis@github.com/derekerdmann/lunch_call.git`
   to: 
   `url=ssh://git@github.com/derekerdmann/lunch_call.git`
   That is, change all the texts before `@` symbol to `ssh://git`
4. Save `config` file and quit. now you could use `git push origin master` to sync your repo on GitHub.



**Use git in colab**

1. connect colab with google drive by command :{from google.colab import drive; drive.mount('/content/drive')}
2. check the file in the folder by command “!ls”
3. change the folder by command “%cd folder path”

4. create a new folder by command “import os ; folderpath =“/”; os.makedirs(folderpath,exist_ok=True”)



###  Offic tools

**PDF**

1. [Compress PDF](https://smallpdf.com/result#r=1d898b1b1d97f6a87d700957b8954afd&t=compress)


### Interpretable AI

**SHAP**
1. [How to use shap](https://blog.csdn.net/sinat_26917383/article/details/115400327)
2. [Draw the figure of shap](https://blog.51cto.com/u_13544/8766623)
