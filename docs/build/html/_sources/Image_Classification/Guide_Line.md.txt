## Tutorial 1 : MNIST

1. Training Phase : See this [link(colab)](https://colab.research.google.com/drive/1cDfsA_no_bLmv5S7217nL0UcEd2nZ1Hr?usp=sharing)

	We trained 50,000 numbers of images(Open Dataset, MNIST).

	You may remember the only ```mnist_99acc_model.h5``` file. (You can [Download here](https://docs.google.com/uc?export=download&id=1tTfB7C4Imavg_ppIwXk0DG-cRkA5P_mt))

	It helps to load model paramters easily. 

	In this course, you can train in colab with GPU for free. It is too hard to train mnist datasets with Jetson Nano toolkits, because of low memories. **I strongly recommend to train with devices having GPUs.**

2. Data Preparation : See this [link(colab)](https://colab.research.google.com/drive/18eUHkOg5jy2YgugphupEjhuAMmem56uD?usp=sharing)
	
	This helps your handmade datasets to become suitable inputs of this model.

	I wrote this codes about Data Preprocessing.

	- How to run? (Your directories must be on ```{your_directory}/AI_beginner_course```)

		1. Prepare only a written number (0 ~ 9)	

		There are some rules. Keep in mind!

		```
		1. The written number should be enough large (Recommend to take 80% of the area)
		2. The written number should be the only one number
		  (e.g. never recognize 32, 64, 58... Only allow "0", "3". etc.)
		3. The more Pen thickness, the better.
		4. The background(paper) should be white. Don't shade the light by cameras.
		```

		For example,
		
		<img src="./3.jpg" width="300px" height="300px">
		
		Then, these result will be

		![index](./index.png)

		2. Run "Data_Preparation.py"
	
		Note that we don't care what the file names are. You only need to any ".jpg" files
		
		At, ```{your_directory}AI_beginner_course/DL_course/}```,
		```shell
		$ tree Image_Classification/
		Image_Classification/
		├── Data_Preperation.py
		├── Test_0.jpg
		├── Test_1.jpg
		├── Test_2.jpg
		└── Test_3.jpg

		0 directories, 5 files
		```
		
		Then ```*.jpg``` files will convert to ```Preprocessed_i.jpg``` files by running this py :
		

		```shell
		python3 Data_Preparation.py
		```

		Results:
		```shell
		Image_Classification/
		├── Data_Preperation.py
		├── PreProcessed
		│   ├── Preprocessed_0.jpg
		│   ├── Preprocessed_1.jpg
		│   ├── Preprocessed_2.jpg
		│   └── Preprocessed_3.jpg
		├── Test_0.jpg
		├── Test_1.jpg
		├── Test_2.jpg
		└── Test_3.jpg

		1 directory, 9 files
		```




3. Inference : See

You can check in ```AI_beginner_course/Image_Classification/Inference.py```.

## Tutorial 2 : Fashion MNIST

We'll introduce Later
