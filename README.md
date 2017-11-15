# Character Finder

## Prerequisites 
* Make sure you have the prerequisites for the Object Detection API installed. The directions for installations can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
* Download the [training](https://www.dropbox.com/s/linj0vexpsfgju3/characters.zip?dl=1) and [evaluation](https://www.dropbox.com/s/057f3o1zsyd8k26/eval_images.zip?dl=1) images in this directory

Run the  following commands 
```
 # Unzip the contents
 unzip characters.zip
 unzip eval_images.zip
 
 # Change the filneame attribute in the train and eval csvs to point to the correct location of the images
 python change_csv.py
```
**Make sure the images are inside the CharacterFinder/ directory** 

## Generating record files

**The following instuctions only creates training and evaluation records for one character - Santa. If you want to train on more characters, follow these [instructions](#abcd) first**

* After the csv points to the correct location, we can generate the record files
 ``` bash
# Generate training record
python --csv_input train.csv --output_path train.record --label_map_path characters_label_map.pbtext 
``` 
``` bash
# Generate evaluation record
python --csv_input eval.csv --output_path eval.record --label_map_path characters_label_map.pbtext 
```


## Training
For training you need to construct an object-detection training pipeline. 
* You can use any of the config files present in object\_detection/samples/configs/ as basis
* Adjust the number of classes depending on the number of character you are training on
* It is recommended to train your model from a pre-trained checkpoint. Tensorflow provides several pre-trained checkpoints which can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
* Changes in the config file:
	1. Change the ` fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt" ` to point to the checkpoint you want to use 
	2. 
		```
		train_input_reader: {
		  tf_record_input_reader {
		    input_path: "PATH_TO_BE_CONFIGURED/train.record"
		  }
		  label_map_path: "PATH_TO_BE_CONFIGURED/characters_label_map.pbtxt"
		}
		```
<a name="abcd"></a>
