# CUDA Installation for use with this project.

- Be sure to have the latest Nvidia Grapic Driver and we need to determine your hardware **CUDA Version** with the following command, anotate the exact version for next steps:

    - On Windows, :
        ```bash
        c:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe
        ```

    - On Linux, run:
        ```bash
        nvidia-smi
        ```
- After finding the correct **CUDA Version** for your device in the output of the previous command, please download and install the **Cuda Toolkit** for your **EXACT CUDA Version** from:
[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

- Go to [TensorFlow, CUDA and cuDNN Compatibility](https://punndeeplearningblog.com/development/tensorflow-cuda-cudnn-compatibility/) and search for the following for your current **CUDA Version** and anotate the versions for the next steps:

    - The **Tensorflow Version**
    - The **Python Version**
    - The **CUDNN Version**

- Search, download and install the correct **CuDNN Version** from the [CuDNN Archive](https://developer.nvidia.com/cudnn-archive) by using the [CuDNN installation instructions](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html)

- Restart your CLI, console or terminal, so the enviroment variables set by CuDNN installation are loaded

- After restarting your console to load the environment variables, activate your conda environment again:
        ```bash
        conda activate feature-extractor-env
        ```
- (Optionally) Update to the required **Python Version** for your **CUDA Version** in your conda environment:

    ```bash
    conda install python=<REQUIRED_PYTHON_VERSION>
    python --version
    ```

- Modify the requirements.txt file to show **tensorflow-gpu==<REQUIRED_TENSORFLOW_VERSION_HERE>** instead of just **tensorflow**, and if using tensorflow-gpu version more than 2.0, remove the **keras** line, since tensorflow-gpu > 2.0, already includes keras-gpu. Save the changes.

- Install the modified **requirements.txt**, this time with **tensorflow-gpu** (Keras-gpu included) instead of just **keras** (you may need to fix some package versions in the readme for the requirements of your current tensorflow-gpu version, if some error appears):

    ```bash
    pip uninstall -y numpy scipy pandas tensorflow keras
    pip install -r requirements.txt --no-cache-dir 
    ```

- Since tensorflow-gpu version 2.0, the keras-gpu package comes included and do not need separate installation, for previous versions, install the keras package with: pip install keras

- To test if Keras is using the GPU:

    ```bash
    python
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()
    exit()
    ```
- If the previous test is passed, the GPU can be used, and no other changes in this repo code are required since it detects if a gpu is available automatically for training and evalation of trained models.