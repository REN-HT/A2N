# A2N   
## architecture  
paper：https://arxiv.org/abs/2104.09497  
![image](https://github.com/REN-HT/A2N/blob/main/images/A2N.jpg)    
![image](https://github.com/REN-HT/A2N/blob/main/images/A2B.jpg)     
## implementation  
1. train.py for train，you can run the file alone  
2. main.py for test，you can run the file alone 
3. change the path when you try to load data  
4. you need to change some parameters in AAN.py and DataSet.py for different scale 
## train   
![image](https://github.com/REN-HT/A2N/blob/main/images/aan_L1_2x_400.jpg)   
说明：基于L1Loss的2倍训练曲线，400个epoch,验证集为div2k100张验证集其中5张，每张裁剪5张，一共25张组成。  
## result
![image](https://github.com/REN-HT/A2N/blob/main/images/psnr.jpg)    
## display  
### 2x  
![image](https://github.com/REN-HT/A2N/blob/main/images/2x.png)  
### 4x  
![image](https://github.com/REN-HT/A2N/blob/main/images/4x.png)  
