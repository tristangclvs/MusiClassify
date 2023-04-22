# MusiClassify

MusiClassify tends to provide a way to discover the music genre of a given music file.
In fact, thanks to a Convolutional Neural Network (CNN) and audio processing libraries, some features are extractable in order to classify genres.

## Dataset 

The dataset used to train this model is the GTZAN Dataset. This dataset consists in 1000 audio files of 30 seconds each, divided in 10 different genres : 
<div align="center" >
<table>
  <tr>
    <th>Blues</th>
    <th>Classical</th>
    <th>Country</th>
    <th>Disco</th>
    <th>Hip-hop</th>
    <th>Jazz</th>
    <th>Metal</th>
    <th>Pop</th>
    <th>Reggae</th>
    <th>Rock</th>
  </tr>
</table>
</div>

## Usage

Due to the large files brought by the previously cited dataset, and the files that resulted in the data processing, the code isn't usable in the state of this repository. 
However, a website will soon be at your disposal to try and find the genre of your most loved musics.

## Matrix

The efficiency of the latest model in date can be displayed by this confusion matrix:

<div align="center" >
  
![Confusion Matrix](https://media.discordapp.net/attachments/806147253504573520/1099279639395971122/model_15_conf.png?width=363&height=325)
  
</div>

## ⛏️ Built Using <a name = "built_using"></a>
- [Python3](https://www.python.org/download/releases/3.0/) - Programming Language
- [TensorFlow](https://www.tensorflow.org/) - Machine/Deep Learning framework


