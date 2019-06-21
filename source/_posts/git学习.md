---
title: Git学习——使用Git将本地库的内容推送到Github
date: 2018-05-27 11:03:13
tags: 
- git
categories:
- 一些摸索
comments: true
---
## 前言

本文参考廖雪峰老师的教程，记录git将本地库的内容推送到Github远程仓库的整个流程。比如我要推送的是叫assignment1的文件夹，内容为斯坦福大学CS231n课程的第一个作业。
## 流程

1. 进入包含assignment1文件夹的目录，把当前目录变成Git可以管理的仓库:

   ```shell
   $ git init
   ```
   <!-- more -->

2. 配置用户信息:

   ```shell
   $ git config --global user.name "xxx"
   $ git config --global user.email "xxx@xxx.com"
   ```

3. 将文件添加到暂存区:

   ```shell
   $ git add assignment1
   ```

4. 提交文件到分支:

   ```shell
   $ git commit -m "add assignment1"
   ```

   双引号里面是本次提交的说明。输入说明对自己和对别人的阅读都很重要，所以建议写上。

5. 在github上建立远程仓库，仓库的名字取为CS231n。

6. 现在，在本地的仓库运行以下命令： 

   ```shell
   $ git remote add origin git@github.com:xxx/CS231n.git
   ```

   这里的xxx替换成自己的Github用户名。

7. 将本地库的所有内容推送到远程库上：

   ```shell
   $ git push -u origin master
   ```

8. 由于远程库是空的，我们第一次推送master分支时，加上了-u参数，Git不但会把本地的master分支内容推送的远程新的master分支，还会把本地的master分支和远程的master分支关联起来，在以后的推送或者拉取时就可以简化命令：

   ```shell
   $ git push origin master
   ```

9. 如果我们对本地库的文件进行了修改，提交到远程仓库只需要进行第3,4以及8步即可。

















