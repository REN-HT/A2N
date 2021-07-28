# A2N   
## architecture  
![image](https://github.com/REN-HT/A2N/blob/main/images/A2N.jpg)    
![image](https://github.com/REN-HT/A2N/blob/main/images/A2B.jpg)     
## implement  
1. train.py文件用于训练，可单独执行  
2. main.py文 用于测试，可单独执行  
3. 注意数据载入路径更改  
4. 不同超分倍数需要更改AAN.py模型中scale参数和DataSet.py相关参数  
## train   
![image](https://github.com/REN-HT/A2N/blob/main/images/aan_L1_2x_400.jpg)   
说明：基于L1Loss的2倍训练曲线，400个epoch,验证集为div2k100张验证集其中5张，每张裁剪5张，一共25张组成。  
## result
![image](https://github.com/REN-HT/A2N/blob/main/images/res.jpg)   
## display  
### 2x  
![image](https://github.com/REN-HT/A2N/blob/main/images/2x.png)  
### 4x  
![image](https://github.com/REN-HT/A2N/blob/main/images/4x.png)  
