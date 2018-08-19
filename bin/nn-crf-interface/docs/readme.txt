## Installation of Neural Network Binding

Ensure you have Java 8:

```
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer
```

You will also need Java binding for ZeroMQ (JZMQ):

Using Homebrew:
```
brew install zmq
```

Install manually:

```
git clone https://github.com/zeromq/jzmq
cd jzmq/jzmq-jni
./autogen.sh && ./configure && make
sudo make install
```

Make sure you have Torch installed:

```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```

and then install Torch dependency libraries:

```
luarocks install lzmq
luarocks install lua-messagepack
luarocks install luautf8
```


## Word embedding

Below is the list of word embedding supported by this package:

polyglot:
Download the .pkl files from [Polyglot]( https://sites.google.com/site/rmyeid/projects/polyglot#TOC-Download-the-Embeddings).
Put these files in `neural_server/polyglot`, then run the following to preprocess for Torch:

```
python pkl2txt.py polyglot-en.pkl > polyglot-en.txt
th bintot7.lua polyglot-en.ttxt polyglot-en.t7
```

Run a neural net server that listens on port 5556 and specify the `gpuid` (>= 0 for GPU, -1 for CPU)

```
th server.lua -port 5556 -gpuid -1
```

Then configure the neural.config accordingly to run your desired neural network.