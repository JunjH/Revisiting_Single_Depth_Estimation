# Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries
<br>
Junjie Hu, Mete Ozay, Yan Zhang, Takayuki Okatani https://arxiv.org/abs/1803.08673

Results
-
![](https://github.com/junjH/Revisiting_Single_Depth_Estimation/raw/master/examples/example.png)
![](https://github.com/junjH/Revisiting_Single_Depth_Estimation/raw/master/examples/results.png)


Dependencies
-
+ python 2.7<br>
+ Pytorch 0.3.1<br>

Running
-

Download the trained models:
[Depth estimation networks](https://drive.google.com/file/d/1QaUkdOiGpMuzMeWCGbey0sT0wXY0xtsj/view?usp=sharing) <br>
Download the data:
[NYU-v2 dataset](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing) <br>
+ ### Demo<br>
  python demo.py<br>
+ ### Test<br>
  python test.py<br>
+ ### Train<br>
  python train.py<br>

Citation
-
If you use the code or the pre-processed data, please cite:

    @inproceedings{Hu2019RevisitingSI,
      title={Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps With Accurate Object Boundaries},
      author={Junjie Hu and Mete Ozay and Yan Zhang and Takayuki Okatani},
      booktitle={2019 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2019}
    }
