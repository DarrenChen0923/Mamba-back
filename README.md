# Mamba-back
# SPIF (Single Point Increment Forming) Prediction Project

This project uses artificial intelligence technology to predict springback errors for SPIF.

The entire project is based on pytorch and scikit-learn libraries.

## Install Package

The code was tested under Ubuntu 24.04.1 LTS, with python 3.9.20.


1. Install dependencies:
```bash
pip install -r requirements.txt
```



## Download Dataset

We have split our data into `training data` and `testing data`, and provide the direct download link, you can downlaod them by the following link:

Dataset: [https://drive.google.com/drive/folders/1Z_MjxmLYjO02014I_9nKFBjvAHb5Ru6P](https://drive.google.com/drive/folders/1Z_MjxmLYjO02014I_9nKFBjvAHb5Ru6P)



Put them into the default main folder, the final directory should look like the following:

```
\models
\utils
\data
    \5mm_file
        \outfile1
            \trainingfile_5mm_overlapping_3.txt
        \outfile2
        \outfile3
   \10mm_file
        \outfile1
            \trainingfile_10mm_overlapping_3.txt
        \outfile2
        \outfile3
   \15mm_file
        \outfile1
            \trainingfile_15mm_overlapping_3.txt
        \outfile2
        \outfile3
   \20mm_file
        \outfile1
            \trainingfile_20mm_overlapping_3.txt
        \outfile2
        \outfile3

```

These would be enough to reproduce the result. 




## Model
Attention-SSM Network Architecture for Springback Prediction. The architecture begins with raw point cloud data divided into grids, flattened into sequences, and processed through linear mapping and positional encoding. A multi-layer self-attention encoder captures contextual dependencies, and the Mamba module refines features, producing final springback error predictions through mean pooling and a fully connected layer.

![image](https://github.com/user-attachments/assets/b647ec64-3783-4a5c-8ac7-fc5f902e56d5)



## Training

To train the model for 

```bash
python train_model.py --project_root <project_root> --grid <5,10,15,20>  --d_model 128
```

The model will be saved into the `trained_models` folder.


## Evaluation

To evaluate the model trained for a certain grid size, run the following:

```
python evalution_model.py \\
--project_root <project_root> \\
--grid <5,10,15,20> \\
--load_model <model_name>
--d_model 128
```

The <model_name> should be specified as the name of one of the models in the `trained_models` folder.

## Checkopoints

We provide the checkpoints for the trained models, feel free to test them!

| Grid Size | Model |
| :---         |     :---:      | 
| Attention-SSM_5mm   | [download](https://drive.google.com/file/d/1-F6FZNlBPT-r9-tp0BDtZTWIuBWn8kPC)    | 
| Attention-SSM_10mm  | [download](https://drive.google.com/file/d/1PQHvJDKHV7wjr6-lAwJGjXWkj-zRmf54)    |
| Attention-SSM_15mm  | [download](https://drive.google.com/file/d/1wrefhVhlC-NmMEeYTommfBZXXiqxIGhi)   | 
| Attention-SSM_20mm  | [download](https://drive.google.com/file/d/1Ko2JMlnlpb_FofLULBAXQDBr3zfLZ-wW)    |


## Result
Key observations include:
1) Attention-SSM Network consistently outperforms all baseline models across all grid sizes.
2) For the 5mm grid size, Attention-SSM Network achieves an MAE of 0.2281 and R² of 0.9421, significantly surpassing LSTM (MAE: 0.3067, R²: 0.9140).
3) As the grid size increases, Attention-SSM Network maintains strong performance, achieving an MAE of 0.2117 and R² of 0.9451 at 20 mm, highlighting its ability to capture structural dependencies effectively and scale efficiently to larger grid intervals.

![image](https://github.com/user-attachments/assets/5c585f19-f3af-4b9c-8261-418aaaea78cd)


Ablation Study on the Impact of Different Mamba Embedding Size

![image](https://github.com/user-attachments/assets/210602c7-7a6e-4c23-aa57-2aba7ae165bc)

Ablation Study on the Impact of Attention Mechanism

![image](https://github.com/user-attachments/assets/3fa6f3c2-1908-4adc-b86b-5545a0db236f)

