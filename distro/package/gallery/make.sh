#!/bin/bash

set -e
PKG=gallery
#QMAKE=/usr/bin/qmake
QMAKE=$TOP_DIR/buildroot/output/rockchip_rk3399/host/bin/qmake
mkdir -p $BUILD_DIR/$PKG
cd $BUILD_DIR/$PKG
$QMAKE $TOP_DIR/app/$PKG
make
mkdir -p $TARGET_DIR/usr/local/$PKG
cp $TOP_DIR/app/$PKG/conf/* $TARGET_DIR/usr/local/$PKG/
install -m 0755 -D $BUILD_DIR/$PKG/galleryView $TARGET_DIR/usr/local/$PKG/galleryView
cd -

