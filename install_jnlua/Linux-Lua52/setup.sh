CURRENT_DIR=$PWD
STATNLP_DIR=$PWD/../..
mkdir -p $STATNLP_DIR/nativeLib
rm -rf jnlua-1.0.4*
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/jnlua/jnlua-1.0.4-src.zip
unzip jnlua-1.0.4-src.zip
cp Makefile jnlua-1.0.4/src/main/c/Linux/Makefile 
cd jnlua-1.0.4/src/main/c/Linux && make
cd $CURRENT_DIR
cp jnlua-1.0.4/src/main/c/Linux/libjnlua5.2.so $STATNLP_DIR/nativeLib/libjnlua52.so
cp $HOME/torch/install/lib/libTH.so $STATNLP_DIR/nativeLib
luarocks install torch
## luarocks install nn
luarocks install rnn
