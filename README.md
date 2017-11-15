# Character Finder

## Prerequisites 
* Make sure you have the prerequisites for the Object Detection API installed. The directions for installations can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

### For Santa Claus (For other characters follow these [instructions](#abcd))
* Download the [training](https://www.dropbox.com/s/c8tbm4obfdupqgs/santa.zip?dl=1) in character-finder/characters directory  and [evaluation](https://www.dropbox.com/s/xij9f2r1wzksfso/santa.zip?dl=1) images in the characterfinder/eval\_image directory
** This only downloads the images for Santa Claus.

Run the following commands 
```
 # Unzip the contents
 # From character-finder/characters
 unzip santa.zip  
 # From character-finder/eval_images
 unzip santa.zip
 
 # Change the filneame attribute in the train and eval csvs to point to the correct location of the images
 # From character-finder/
 python change_csv.py
```
<a name="abcd"></a>

<details>
 <summary><h3>Training on Other Characters</h3></summary>
 <p>Will ad some information here</p>
</details>

## Generating record files


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
* Changes in the .config file:
	1. Change the ` fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt" ` to point to the checkpoint you want to use 
	2. In the following Code snippet
		```
		train_input_reader: {
		  tf_record_input_reader {
		    input_path: "PATH_TO_BE_CONFIGURED/train.record"
		  }
		  label_map_path: "PATH_TO_BE_CONFIGURED/characters_label_map.pbtxt"
		}
		```
		Change the ` input_path: "PATH_TO_BE_CONFIGURED/train.record" ` to point to the train.record file created in the previous step and the ` label_map_path: "PATH_TO_BE_CONFIGURED/characters_label_map.pbtxt" ` to point to the appropriate label map
	3. In the following Code snippet
		```
		eval_input_reader: {
  		  tf_record_input_reader {
    		    input_path: "PATH_TO_BE_CONFIGURED/eval.record"
  		  }
  		  label_map_path: "PATH_TO_BE_CONFIGURED/characters_label_map.pbtxt"
  		  shuffle: false
  		  num_readers: 1
  		  num_epochs: 1
		}
		```
		Change the ` input_path: "PATH_TO_BE_CONFIGURED/eval.record" ` to point to the eval.record file created in the previous step and the ` label_map_path: "PATH_TO_BE_CONFIGURED/characters_label_map.pbtxt" ` to point to the appropriate label map

<h3>Hello</h3>
