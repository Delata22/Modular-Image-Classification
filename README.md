# Modular Image Classification Framework by Gurvan Estable

This project provides a robust and flexible architecture for training and using image classification models. Built on Docker, PyTorch Lightning, and Hydra, it is designed to be highly modular, allowing you to experiment with different datasets and model architectures with minimal friction.

---

## Key Features

* **Isolated & Reproducible Environment**: Fully containerized with Docker, ensuring the code runs the same way on any machine.
* **Flexible Configuration**: Driven by [Hydra](https://hydra.cc/), allowing you to change datasets, models, and hyperparameters through simple command-line arguments.
* **Automatic Data Preparation**: A script inspects your image folders, discovers classes, and prepares the data for efficient training.
* **Model Modularity**: Load any image classification model from the Hugging Face Hub by changing a single parameter.
* **Comprehensive Evaluation**: Automatically generates test reports at the end of training, including a confusion matrix and a list of classification errors.

---

## Prerequisites

Before you begin, ensure you have installed:

1.  **Docker & Docker Compose**: [Docker installation instructions](https://docs.docker.com/get-docker/).
2.  **(Optional, for GPU training)** An NVIDIA GPU with up-to-date [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## Quick Start

### 1. Folder Structure

Ensure your project follows this structure:

```text
.
├── configs/                        # Configuration files
├── data/                   
|   ├── metadata                    # [TRAIN ONLY] Generated meatada folder
│   └── raw_data/                   # [TRAIN ONLY] Datasets folder
├── input/                          # [INFERENCE ONLY] This folder is where you will put your images to use the model on
├── models/                         # [TRAIN ONLY] This folder is where the models will be stored after training
├── out/                            # [INFERENCE ONLY] This folder is where the result of inference will be stored as json files
├── src/                            # Source code
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

### 2. Build and Run the Environment

Open a terminal at the project root and run the following commands:

```bash
# 1. Build the Docker image (only needed once or after modifying requirements.txt)
docker compose build

# 2. Start the container in the background
docker compose up -d
```


Your development environment is now ready and running.

---

## [TRAINING]: Training a New Model

The process only consists of one simple command.

### Step 1: Prepare the Dataset

Place your images in `data/raw_data/` following the `DATASET_NAME/CLASS_NAME/images...` structure.

**Example:**

With a dataset called EXAMS of two classes (good, bad), the architecture will look like this :

```text
└── data/                   
    └── raw_data/                  
        └── EXAMS/             
            ├── good          
            |   ├── good_image_1.jpg  
            |   ├── ...
            |   └── good_image_n.jpg 
            └── bad
                ├── bad_image_1.jpg  
                ├── ...
                └── bad_image_n.jpg  
```

### Step 2: Run the Training Command

Launch the training by specifying your dataset's name. The script will first check if the data has already been prepared in the data/metadata/DATASET_NAME.  If not, it will prepare it before starting the training.

**Exemple:**

```bash
docker exec modular-ai-app python -m src.train ds=EXAMS
```

If you changed the container name in the docker-compose file, change modular-ai-app by the new name

### Step 3: Verify the results

After training, your model will be stored in **models/DATASET_NAME/default/YYYY-MM-DD/hh-mm-ss**.
Here, you can access usefull informations about the training such as the classifications errors and the confusion matrix in the **train_info** directory

You can also check the performance of the model on the validation set by looking at the name of the checkpoint file in the **checkpoints** directory.

Finally, you can use tensorboard to have more in-depth evaluation on the models performance:

1.  Start the TensorBoard server: `docker exec -d modular-ai-app tensorboard --logdir models --bind_all`
2.  Open your browser and navigate to **`http://localhost:6006`**. Here you can watch your model's performance (like `val_acc` and `train_loss`) update in real-time.

## [INFERENCE]: Testing an existing Model

### Step 1: set up the images

First of all, put the images you want to use the model on in the **input/** folder.

### Step 2: retrieve the Model

After training, you will see at the end of the log a line like this:

```bash
"Your model as been saved in the models/DATASET/CATEGORY/YY-MM-DD/hh-mm-ss directory"
```

Copy the path said directory

### Step 3: launch the inference

Write the following command:

```bash
docker exec modular-ai-app python -m src.predict md=models/DATASET/CATEGORY/YY-MM-DD/hh-mm-ss
```

Although not recommended, you can also directly put the model path in the **md** variable in the **configs/predict.yaml** file and simply write :

```bash
docker exec modular-ai-app python -m src.predict
```

Note that the GPU will automatically be used if available, but is not necessary for inference.

### Step 4: get the results

For each image in the **input/** folder, a corresponding json file will be created, with the top 3 predictions of the AI, their probabilities and the path of the original image.


## [TRAINING] Advanced customization

### Model category

By default, your model will be saved in the following directory:

**models/DATASET_NAME/default/YYYY-MM-DD/hh-mm-ss** 

But you can specify a category in the command-line in order to save it in an another subdirectory of **models/DATASET_NAME** like so:

```bash
docker exec modular-ai-app python -m src.train ds=EXAMS category=my_category
```

This model will be saved in 

**models/DATASET_NAME/my_category/YYYY-MM-DD/hh-mm-ss** 

### Model customization

You can change the model used for the training by changing the variable **model_name** in **configs/model/default.yaml**.
You can also specify the model directly in the command-line :

```bash
docker exec modular-ai-app python -m src.train ds=EXAMS model.model_name=NEW_MODEL
```

You can chose any model from [Hugging Face](https://huggingface.co/models) as long as it comes from the **Image Feature Extraction** or the **Image Classification** category.

The default model is **google/vit-base-patch16-224-in21k** for its good performance and training speed.

### Hyperparameters customization

Similarly to the model customization, you can change the learning rate by changing the variable **lr** in **configs/model/default.yaml**.
Or in the command-line:

```bash
docker exec modular-ai-app python -m src.train ds=EXAMS model.lr=NEW_LR
```

### CPU training

While it is not advised due to the low speed of CPUs, you can train a model on CPU.
To do so, simply modify the defaults.trainer value in **configs/callbacks/default.yaml** to "cpu"

Or in the command-line:

```bash
docker exec modular-ai-app python -m src.train ds=EXAMS trainer=cpu
```

### Training strategy

By default, the training strategy consists of tracking the accuracy of the validation set **val_acc** and continuing the training while this value increases.
If this value stops increasing for 3 epochs in a row, the training will be stopped. 
This behavior is defined in **configs/callbacks/default.yaml** and can be disabled or modified, by deleting and modifying the **early_stopping** section respectively.

The maximum number of epochs is defined in **configs/trainer/gpu.yaml** (or **configs/trainer/cpu.yaml** if you are using cpu) and can be modified.

### Checkpoint strategy

By default, the training process will select the best performing version of the model based on the **val_acc** metric.
To do so, at every epoch, the current **val_acc** will be computed and compared to the **val_acc** of the previous best model. 
The new model will replace the old one if the new model performed better and be saved in the **models/DATASET_NAME/default/YYYY-MM-DD/hh-mm-ss/checkpoints** directory, in .ckpt format.

You can change the checkpoint strategy in the **configs/callbacks/default.yaml** file


## [INFERENCE] Advanced configuration

### Single file prediction

If you want to make a prediction of a single file without moving it to the **input/** folder, you can do so by specifying its path in the command like this:

```bash
docker exec modular-ai-app python -m src.predict md=models/DATASET/CATEGORY/YY-MM-DD/hh-mm-ss file=FILE_PATH
```

In which case the **input/** folder will not be used for prediction.
Like before, you can also directly put the filepath in the **configs/predict.yaml** file and remove the file argument from the command :

```bash
docker exec modular-ai-app python -m src.predict md=models/DATASET/CATEGORY/YY-MM-DD/hh-mm-ss
```

### Model sharing

You can directly share a trained model to someone else possessing this script by copying the **models/DATASET/CATEGORY/YY-MM-DD/hh-mm-ss** folder.
Then the receiver can put it back where he wants in the **models** folder, renaming it if he wants to, and use the following command:

```bash
docker exec modular-ai-app python -m src.predict md=models/MODEL_PATH
```

