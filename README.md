# A2N   
## Blog
[zhihu](https://www.zhihu.com/people/longing-93-89/posts "Brand·R")<br>
## architecture
paper：https://arxiv.org/abs/2104.09497<br>
![image](https://github.com/REN-HT/A2N/blob/main/images/A2N.jpg)<br>  
![image](https://github.com/REN-HT/A2N/blob/main/images/A2B.jpg)<br>
## implementation
1. train.py for train，you can run the file alone<br>
2. main.py for test，you can run the file alone<br>
3. change the path when you try to load data<br>
4. you need to change some parameters in AAN.py and DataSet.py for different scale<br>
## train
![image](https://github.com/REN-HT/A2N/blob/main/images/aan_L1_2x_400.jpg)<br>
validation set: select 5 images from div2k 100 validation set, then clipping them to 25 images<br>
## result
![image](https://github.com/REN-HT/A2N/blob/main/images/psnr.jpg)<br>
## display
### 2x
![image](https://github.com/REN-HT/A2N/blob/main/images/2x.png)<br>
### 4x
![image](https://github.com/REN-HT/A2N/blob/main/images/4x.png)<br>
