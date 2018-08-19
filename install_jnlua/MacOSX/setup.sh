CURRENT_DIR=$PWD
STATNLP_DIR=$PWD/../..
mkdir -p $STATNLP_DIR/nativeLib
rm -rf jnlua-0.9.6*
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/jnlua/jnlua-0.9.6-src.zip
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/jnlua/jnlua-0.9.6-bin.zip
unzip jnlua-0.9.6-src.zip
cp Makefile jnlua-0.9.6/src/main/c/MacOSX/Makefile 
cd jnlua-0.9.6/src/main/c/MacOSX && make
cd $CURRENT_DIR
cp jnlua-0.9.6/src/main/c/MacOSX/libjnlua5.1.jnilib $STATNLP_DIR/nativeLib
cp $HOME/torch/install/lib/libTH.dylib $STATNLP_DIR/nativeLib
unzip jnlua-0.9.6-bin.zip
luarocks install torch
luarocks install nn
luarocks install rnn
