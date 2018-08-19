CURRENT_DIR=$PWD
STATNLP_DIR=$PWD/../..
mkdir -p $STATNLP_DIR/nativeLib
rm -rf jnlua-1.0.4*
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/jnlua/jnlua-1.0.4-src.zip
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/jnlua/jnlua-1.0.4-bin.zip
unzip jnlua-1.0.4-src.zip
cp Makefile jnlua-1.0.4/src/main/c/MacOSX/Makefile 
cd jnlua-1.0.4/src/main/c/MacOSX && make
cd $CURRENT_DIR
cp jnlua-1.0.4/src/main/c/MacOSX/libjnlua5.2.jnilib $STATNLP_DIR/nativeLib/libjnlua52.jnilib
cp $HOME/torch/install/lib/libTH.dylib $STATNLP_DIR/nativeLib
luarocks install torch
## luarocks install nn
luarocks install rnn
