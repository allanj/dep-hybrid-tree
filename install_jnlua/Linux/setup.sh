CURRENT_DIR=$PWD
STATNLP_DIR=$PWD/../..
mkdir -p $STATNLP_DIR/nativeLib
rm -rf jnlua-0.9.6*
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/jnlua/jnlua-0.9.6-src.zip
unzip jnlua-0.9.6-src.zip
cp Makefile jnlua-0.9.6/src/main/c/Linux/Makefile 
cd jnlua-0.9.6/src/main/c/Linux && make
cd $CURRENT_DIR
cp jnlua-0.9.6/src/main/c/Linux/libjnlua5.1.so $STATNLP_DIR/nativeLib
cp $HOME/torch/install/lib/libTH.dylib $STATNLP_DIR/nativeLib
luarocks install torch
luarocks install nn
luarocks install rnn
