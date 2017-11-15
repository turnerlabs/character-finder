# Character Finder

## Prerequisites 
* Make sure you have the prerequisites for the Object Detection API installed. The directions for installations can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

<details>
<summary>For Santa Claus</summary>

* Download the [training](https://www.dropbox.com/s/c8tbm4obfdupqgs/santa.zip?dl=1) in character-finder/characters directory  and [evaluation](https://www.dropbox.com/s/xij9f2r1wzksfso/santa.zip?dl=1) images in the characterfinder/eval\_image directory
**This only downloads the images for Santa Claus.**

Run the following commands 
```
 # Unzip the contents
 # From character-finder/characters
 unzip santa.zip  
 # From character-finder/eval_images
 unzip santa.zip
 
 # From character-finder/
 python change_csv.py
 # This will change the path in the already existing csvs to point to the images in the correct directory
 
```
</details>
<details>
<summary>For Other Characters</summary>

* Make a new directory in the character-finder/characters folder and name it the character. For example <br>
	` mkdir characters/foobar ` <br> Similarly do the same for the evaluation images <br>
	` mkdir eval_images/foobar ` <br>

* Save all training images for that character in `characters/foobar` and the evaluation images in `eval\_images/foobar`
* Next step is to get bounding box imformation about the characters and store it in a csv which will later be converted to tf.record file. Done for both training and evaluation images.
* Run the following command
	` python detect_labels.py --annotation_file PATH_TO_CSV --images characters/ `
**Note: Already existing train.csv and eval.csv have bounding box information for images of Santa Claus**
* Label the images by clicking on the top left of the characters face first and then on the bottom right
<br>
![](https://media.giphy.com/media/xUNd9BNT18JAOzc0wM/giphy.gif)
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


