<p align="center">
<img src="https://imghost.ipv4.host/d/v7VJdXrW/2022/11/08/DkiLSXZN/icon180.png?download=1" width = "180" alt="pixivic_icon"/>
</p>


## Introduction

**ACG2vec**全称为**A**nime **C**omics **G**ames **to** **vec**tor。本repo会持续维护一些基于二次元相关的深度学习领域实践与探索。

目前模块包括：

* model：深度神经网络模型模块，目前包括
  * **acgvoc2vec**：基于从维基百科动漫列表、萌娘百科、Bangumi、pixiv、AnimeList等来源获取清洗处理抽取的510w语句对微调的sentence-transformers模型，生成二次元相关文本的特征向量，用于各种下游任务（标签推荐，标签搜索，推荐系统等）
  
    可以使用Huggingface在线体验:https://huggingface.co/OysterQAQ/ACGVoc2vec
  
  * **deepix**：基于[DeepDanbooru](https://github.com/KichangKim/DeepDanbooru)模型中抽取activation_96及之前的layer作为encoder(lr:1e-5)，拼接自定义的resnet block与预测头(lr:1e-2)的对pixiv数据进行多任务预测的模型
  
  * **illust2vec**：从deepix去除自定义resnet block与预测头的图片特征抽取模型
  
* webapp：对外提供web服务模块。目前包括开箱即用的二次元插画标签预测服务、以图搜图服务、插画特征抽取服务、文本特征抽取服务

* docker：基于容器化的部署模块，包括了部署所需要的配置文件与资源文件

## Architecture

<img src="https://raw.githubusercontent.com/OysterQAQ/Blog-Image/master/arch.png" alt="image-20220827172516288" style="border-radius:10px" />

## Model Structure

### illust2vec

结构概览为：[DeepDanbooru](https://github.com/KichangKim/DeepDanbooru)的输入层至activation_96层作为特征抽取器（学习率设置为1e-5）+各个任务自定义resnet block与dense预测头（学习率设置为1e-2）

预测任务为pixiv插画的浏览数、收藏数、图片浏览级别（文本标签可以使用DeepDanbooru模型进行预测）。

DeepDanbooru模型是基于resnet的预测模型，用于预测动漫插画的标签信息，完整模型输出纬度为8000。DeepDanbooru能很好的预测Danbooru数据集所描述的多标签多分类问题，Danbooru数据集的标签分布更加的均衡，对图片的描述更加的准确，但是标签中没有对图片收藏数与浏览数的预测，因此其输出中并没有包含图片的质量信息（一般笔触细腻，作画精美的作品会得到更多的浏览与收藏）。

因此考虑将DeepDanbooru的前半部分拼接上自定义任务的模块，以预测收藏浏览作为代理任务的方式对DeepDanbooru的前半部分进行小学习率的微调（自定义任务的模块使用正常学习率），使其能够包含插图的质量信息。

当模型拟合，将模型的特征抽取器模块单独取出，拼接上平均pooling层，使其做到输入一张图片，输出1024维的向量，该向量作为图片的特征向量，用于下游任务。

之前也考虑过使用Danbooru数据集微调CLIP，但是loss一直不变，大概原因可能是这种对比学习的模型，其学习效率与batchsize相关性很大，batchsize越大，正样本所对应的负样本就越多。

### acgvoc2vec

结构为[sentence-transformers](https://github.com/UKPLab/sentence-transformers)，使用其**distiluse-base-multilingual-cased-v2**预训练权重，以5e-5的学习率在动漫相关语句对数据集下进行微调，损失函数为MultipleNegativesRankingLoss。

数据集主要包括：

* Bangumi

  * 动画日文名-动画中文名
  * 动画日文名-简介
  * 动画中文名-简介
  * 动画中文名-标签
  * 动画日文名-角色
  * 动画中文名-角色
  * 声优日文名-声优中文名

* pixiv

  * 标签日文名-标签中文名
* AnimeList

  * 动画日文名-动画英文名

* 维基百科

  * 动画日文名-动画中文名
  * 动画日文名-动画英文名
  * 中英日详情页h2标题及其对应文本
  * 简介多语言对照（中日英）
  * 动画名-简介（中日英） 

* moegirl

  * 动画中文名的简介-简介
* 动画中文名+小标题-对应内容

在进行爬取，清洗，处理后得到510w对文本对（还在持续增加），batchzise=80训练了20个epoch，使st的权重能够适应该问题空间，生成融合了领域知识的文本特征向量（体现为有关的文本距离更加接近，例如作品与登场人物，或者来自同一作品的登场人物）。

## Technical overview

* Tensorflow 2.0作为模型训练引擎
* 基于Spring Boot的web服务
* 基于TF-serving的模型部署与前向推理
* 基于Milvus实现的topk近似向量检索
* 基于docker-compose的容器化跨平台部署
* 基于Tendis的元数据存储

## Deploy

克隆本repo并在docker文件夹中使用docker-compose进行部署

```shell
#拉取项目
git clone https://github.com/OysterQAQ/ACG2vec.git
cd ACG2vec/ACG2vec-docker
#下载release中的模型包 解压到docker/tf-serving/models
#使用docker-compose部署
docker-compose up -d
```

## Usage

基于restful api对外提供服务，以下是api文档（默认端口为8081，可在docker-compose.yaml中修改）：

### 获取插图特征向量

#### 基本信息

**Path：** /images/features

**Method：** POST

**接口描述：**


#### 请求参数

**Headers**

| 参数名称     | 参数值                | 是否必须 | 示例 | 备注 |
| ------------ | --------------------- | -------- | ---- | ---- |
| Content-Type | application/form-data | 是       |      |      |
| **Query**    |                       |          |      |      |

| 参数名称 | 是否必须 | 示例 | 备注     |
| -------- | -------- | ---- | -------- |
| file     | 是       |      | 插图文件 |

#### 返回数据

<table>
  <thead class="ant-table-thead">
    <tr>
      <th key=name>名称</th><th key=type>类型</th><th key=required>是否必须</th><th key=default>默认值</th><th key=desc>备注</th><th key=sub>其他信息</th>
    </tr>
  </thead><tbody className="ant-table-tbody"><tr key=0-0><td key=0><span style="padding-left: 0px"><span style="color: #8c8a8a"></span> message</span></td><td key=1><span>string</span></td><td key=2>非必须</td><td key=3></td><td key=4><span style="white-space: pre-wrap"></span></td><td key=5></td></tr><tr key=0-1><td key=0><span style="padding-left: 0px"><span style="color: #8c8a8a"></span> data</span></td><td key=1><span>string</span></td><td key=2>非必须</td><td key=3></td><td key=4><span style="white-space: pre-wrap">token</span></td><td key=5></td></tr>
               </tbody>
              </table>





## Thanks

本项目离不开以下开源项目

* [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru)
* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
* [Milvus](https://github.com/milvus-io/milvus)
* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [Keras](https://github.com/keras-team/keras)
* [Spring Boot](https://github.com/spring-projects/spring-boot)
* [SeaweedFS](https://github.com/seaweedfs/seaweedfs)
* [TF-serving](https://github.com/tensorflow/serving)
* [Tendis](https://github.com/Tencent/Tendis)
* [Docker](https://github.com/docker/compose)

## Trend

![stars](https://starchart.cc/OysterQAQ/ACG2vec.svg)

