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