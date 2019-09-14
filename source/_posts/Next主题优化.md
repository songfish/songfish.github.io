---
title: Next主题的简单优化(一)
date: 2019-09-14 14:19:06
tags: hexo
comments: true
categories: 一些摸索
---

## 前言

之前匆匆忙忙建站，没有加评论、搜索、数据统计与分析、搜索功能等等，这些功能对于搭建博客也是很重要的。参考了很多大佬的博客，受益匪浅，以下是我的一些摸索。

实际上[Next主题的官方文档](https://theme-next.iissnan.com/getting-started.html)非常详细了，建议多查看。

Next主题版本：Muse v6.3.0

## 评论系统

一开始按照[Next主题的官方文档](https://theme-next.iissnan.com/getting-started.html)配置[来必力](https://livere.com/)评论系统，但是后来发现来必力加载速度有点慢。于是转用基于[LeanCloud](https://leancloud.cn/dashboard/login.html#/signup)的评论系统Valine，Valine也是有[官方文档](https://valine.js.org/)的（看官方文档可是个好习惯）。

<!--more-->

**简要步骤如下：**

**1.获取APP ID和APP Key**。首先在[LeanCloud](https://leancloud.cn/dashboard/login.html#/signup)注册自己的账号。进入[控制台](https://leancloud.cn/dashboard/applist.html#/apps)创建应用。应用创建好以后，进入刚创建的应用，选择`设置`>`应用Key`，就能看到`APP ID`和`APP Key`了：

![](/Users/songyu/songfish.github.io/source/_posts/Next主题优化/index.png)

**2.设置安全域名 :**

![](/Users/songyu/songfish.github.io/source/_posts/Next主题优化/006qRazegy1fkxqmddfh1j30qd0go40h.png)

**3.修改`主题配置文件`中的Valine部分 :**

（未开邮件提醒​​）

文件位置：`themes/next/_config.yml`

```yaml
# Valine.
# You can get your appid and appkey from https://leancloud.cn
# more info please open https://valine.js.org
valine:
  enable: true
  appid: your APP ID
  appkey: your Key
  notify: false # mail notifier , https://github.com/xCss/Valine/wiki
  verify: false # Verification code
  placeholder: Just go go # comment box placeholder
  avatar: monsterid # gravatar style
  guest_info: nick,mail,link # custom comment header
  pageSize: 10 # pagination size
```

**4.如需取消某个页面/文章 的评论，在 md 文件的 [front-matter ](https://hexo.io/docs/front-matter.html)中增加 `comments: false`。**

## 数据统计与分析

### 文章阅读量统计

1.仍然使用LeanCloud。按下图创建`Class`，`Class`名称必须为`Counter`。

![](/Users/songyu/songfish.github.io/source/_posts/Next主题优化/35529984.png)

2.修改`主题配置文件`中的`leancloud_visitors`配置项：

```yaml
leancloud_visitors:
  enable: true
  app_id: 
  app_key: 
```

### 博客访问量统计

用的是`不蒜子统计`，修改`主题配置文件`中的`busuanzi_count`的配置项，当`enable: true`时，代表开启全局开关。

```yaml
# Show Views/Visitors of the website/page with busuanzi.
# Get more information on http://ibruce.info/2015/04/04/busuanzi/
busuanzi_count:
  enable: true
  total_visitors: true
  total_visitors_icon: user
  total_views: true
  total_views_icon: eye
  post_views: false
  post_views_icon: eye
```

## 博客图标

网站的默认图标不是特别好看，因此换成了现在的小鱼。

**修改方法：**

1.到这个神奇的网站[EasyIcon](http://www.easyicon.net/)找心仪的图标，下载`32PX`和`16PX`的`ICO`格式，并把它们放在`/themes/next/source/images`里。

![](/Users/songyu/songfish.github.io/source/_posts/Next主题优化/95085113.png)

2.修改`主题配置文件`中的`favicon`配置项，其中`small`对应`16px`的图标路径，`medium`对应`32px`的图标路径。

```yaml
favicon:
  small: /images/favicon-16x16.ico
  medium: /images/favicon-32x32.ico
  apple_touch_icon: /images/apple-touch-icon-next.png
  safari_pinned_tab: /images/logo.svg
  #android_manifest: /images/manifest.json
  #ms_browserconfig: /images/browserconfig.xml
```

## 博客运行时间

来源[reuixiy](https://reuixiy.github.io/)的[博客](https://reuixiy.github.io/technology/computer/computer-aided-art/2017/06/09/hexo-next-optimization.html#%E5%A5%BD%E7%8E%A9%E7%9A%84%E5%86%99%E4%BD%9C%E6%A0%B7%E5%BC%8F)。

文件位置：`themes/next/layout/_custom/sidebar.swig`（其中的`BirthDay`改成自己的）

```html
<div id="days"></div>
<script>
function show_date_time(){
window.setTimeout("show_date_time()", 1000);
BirthDay=new Date("05/20/2018 15:13:14");
today=new Date();
timeold=(today.getTime()-BirthDay.getTime());
sectimeold=timeold/1000
secondsold=Math.floor(sectimeold);
msPerDay=24*60*60*1000
e_daysold=timeold/msPerDay
daysold=Math.floor(e_daysold);
e_hrsold=(e_daysold-daysold)*24;
hrsold=setzero(Math.floor(e_hrsold));
e_minsold=(e_hrsold-hrsold)*60;
minsold=setzero(Math.floor((e_hrsold-hrsold)*60));
seconds=setzero(Math.floor((e_minsold-minsold)*60));
document.getElementById('days').innerHTML="已运行"+daysold+"天"+hrsold+"小时"+minsold+"分"+seconds+"秒";
}
function setzero(i){
if (i<10)
{i="0" + i};
return i;
}
show_date_time();
</script>
```

文件位置：`themes/next/layout/_macro/sidebar.swig`  (其中加上带加号的那句)

```html
 {# Blogroll #}
        {% if theme.links %}
          <div class="links-of-blogroll motion-element {{ "links-of-blogroll-" + theme.links_layout | default('inline') }}">
            <div class="links-of-blogroll-title">
              <i class="fa  fa-fw fa-{{ theme.links_icon | default('globe') | lower }}"></i>
              {{ theme.links_title }}&nbsp;
              <i class="fa  fa-fw fa-{{ theme.links_icon | default('globe') | lower }}"></i>
            </div>
            <ul class="links-of-blogroll-list">
              {% for name, link in theme.links %}
                <li class="links-of-blogroll-item">
                  <a href="{{ link }}" title="{{ name }}" target="_blank">{{ name }}</a>
                </li>
              {% endfor %}
            </ul>
+        {% include '../_custom/sidebar.swig' %} 
          </div>
         {% endif %}

```

## 搜索功能

文件位置：`themes/next/_config.yml`

```yaml
# Local search
# Dependencies: https://github.com/theme-next/hexo-generator-searchdb
local_search:
  enable: ture
```

安装插件

```shell
$ npm install hexo-generator-search --save
```

但是我在安装插件的时候一直报错

```shell
npm ERR! path /home/song/hexo/test/node_modules/babylon
npm ERR! code ENOENT
npm ERR! errno -2
npm ERR! syscall access
npm ERR! enoent ENOENT: no such file or directory, access '/home/song/hexo/test/node_modules/babylon'
npm ERR! enoent This is related to npm not being able to find a file.
npm ERR! enoent 

npm ERR! A complete log of this run can be found in:
npm ERR!     /home/song/.npm/_logs/2018-11-11T06_59_34_564Z-debug.log
```

[解决办法](https://blog.csdn.net/h416756139/article/details/50812109)：

```shell
$ npm install -g cnpm --registry=http://registry.npm.taobao.org
$ cnpm install hexo-generator-search --save
```



## 预告

1.关于更新主题

2.关于如何推广博客

3.评论邮件提醒