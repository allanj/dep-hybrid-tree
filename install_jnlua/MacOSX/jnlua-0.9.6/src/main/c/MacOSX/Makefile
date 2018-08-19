# Paths
JDK_DIR=/System/Library/Frameworks/JavaVM.framework
LUA_LIB_DIR=$(HOME)/torch/install/lib
LUA_INC_DIR=$(HOME)/torch/install/include
VERSION=5.1

# Tools
CC=gcc
LD=gcc

# Default arguments
CFLAGS=-c -fno-strict-aliasing -m64 -fPIC -O2 -Wall -DNDEBUG -D_REENTRANT -DLUA_USE_LINUX
LDFLAGS=-dynamiclib -m64

# Description blocks
all: libjnlua$(VERSION).jnilib

libjnlua$(VERSION).jnilib: jnlua.o
	$(LD) $(LDFLAGS) -o libjnlua$(VERSION).jnilib -L$(LUA_LIB_DIR) -lc -llua jnlua.o

jnlua.o: ../jnlua.c
	$(CC) $(CFLAGS) -I$(JDK_DIR)/Headers -I$(LUA_INC_DIR) ../jnlua.c

clean:
	-rm libjnlua$(VERSION).jnilib jnlua.o
