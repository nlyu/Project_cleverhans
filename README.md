# Project_cleverhans

This is a cleverhans based research project by zzhu and nlyu.

Basically we transfrom normal pictures to malicious pictures.

For example, in MNIST dataset a normal zero becomes a zero that cannot be regonized by the previous ML model under fgsm attack:

<img src="https://github.com/nlyu/Project_cleverhans/blob/master/source/zero_good.png" width="60"/>

<img src="https://github.com/nlyu/Project_cleverhans/blob/master/source/zero_bad.png" width="60"/>

### Support producing adversarial image with fgsm model for:

1. 28 * 28 single channel grey image (format: numpyarray shape of [number of image * 28 * 28 * 1] float32)
2. 32 * 32 three channel rgb image (format: numpyarray shape of [number of image * 32 * 32 * 3] float32)

* Notice: All image are matplotlib format, rgb value 0 ~ 1, not 0 ~ 255

### Reference:

German Traffic Sign Dataset:

`http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads`

Dataset for 32 * 32 formated traffic sign dataset:

`https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip`

Label leadking effect, why do we need untarget data explained:

`http://jackhaha363.github.io/blog/2017/06/19/label-leaking`

The basic knowledge and intuition for fgsm and math behinded:

`https://zhuanlan.zhihu.com/p/37260275`

Another usedful repo for producing adv image:

`https://github.com/gongzhitaao/tensorflow-adversarial`
