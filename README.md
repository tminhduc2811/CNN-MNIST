# CNN-MNIST
An example of CNN for Image Classification using MNIST data set

## Model: 

<p align="center">
  <img src="Images/Model.png">
</p>

## Train result:

<p align="center">
  <img src="Images/results.JPG">
</p>

<p align="center">
  <img src="Images/TrainFigures.png">
</p>

## Test:
- To use trained model: 
```
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = models.model_from_json(loaded_model_json)

```
<p align="center">
  <img src="Images/TestImage.JPG">
</p>
