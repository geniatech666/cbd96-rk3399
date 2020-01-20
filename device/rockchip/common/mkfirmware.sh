#!/bin/bash

set -e

COMMON_DIR=$(cd `dirname $0`; pwd)
if [ -h $0 ]
then
        CMD=$(readlink $0)
        COMMON_DIR=$(dirname $CMD)
fi
cd $COMMON_DIR
cd ../../..
TOP_DIR=$(pwd)
RELATIVE_TOP_DIR=..

source $TOP_DIR/device/rockchip/.BoardConfig.mk
ROCKDEV=$TOP_DIR/rockdev
PARAMETER=$TOP_DIR/device/rockchip/$RK_TARGET_PRODUCT/$RK_PARAMETER
OEM_DIR=$TOP_DIR/device/rockchip/oem/$RK_OEM_DIR
USER_DATA_DIR=$TOP_DIR/device/rockchip/userdata/$RK_USERDATA_DIR
MISC_IMG=$TOP_DIR/device/rockchip/rockimg/$RK_MISC
ROOTFS_IMG=$TOP_DIR/$RK_ROOTFS_IMG
RAMBOOT_IMG=$TOP_DIR/buildroot/output/$RK_CFG_RAMBOOT/images/ramboot.img
RECOVERY_IMG=$TOP_DIR/buildroot/output/$RK_CFG_RECOVERY/images/recovery.img
TRUST_IMG=$TOP_DIR/u-boot/trust.img
UBOOT_IMG=$TOP_DIR/u-boot/uboot.img
BOOT_IMG=$TOP_DIR/kernel/$RK_BOOT_IMG
LOADER=$TOP_DIR/u-boot/*_loader_v*.bin
#SPINOR_LOADER=$TOP_DIR/u-boot/*_loader_spinor_v*.bin
MKIMAGE=$TOP_DIR/device/rockchip/common/mk-image.sh
mkdir -p $ROCKDEV

# Require buildroot host tools to do image packing.
if [ ! -d "$TARGET_OUTPUT_DIR" ]; then
    echo "Source buildroot/build/envsetup.sh"
    source $TOP_DIR/buildroot/build/envsetup.sh $RK_CFG_BUILDROOT
fi

if [ $RK_ROOTFS_IMG ]
then
	if [ -f $ROOTFS_IMG ]
	then
		echo -n "create rootfs.img..."
		ln -s -f `echo $ROOTFS_IMG | sed "s;$TOP_DIR;$RELATIVE_TOP_DIR;"` $ROCKDEV/rootfs.img
		echo "done."
	else
		echo "warning: $ROOTFS_IMG not found!"
	fi
fi

if [ -f $PARAMETER ]
then
	echo -n "create parameter..."
	ln -s -f `echo $PARAMETER | sed "s;$TOP_DIR;$RELATIVE_TOP_DIR;"` $ROCKDEV/parameter.txt
	echo "done."
else
	echo "warning: $PARAMETER not found!"
fi

if [ $RK_CFG_RECOVERY ]
then
	if [ -f $RECOVERY_IMG ]
	then
		echo -n "create recovery.img..."
		ln -s -f `echo $RECOVERY_IMG | sed "s;$TOP_DIR;$RELATIVE_TOP_DIR;"` $ROCKDEV/recovery.img
		echo "done."
	else
		echo "warning: $RECOVERY_IMG not found!"
	fi
fi

if [ $RK_MISC ]
then
	if [ -f $MISC_IMG ]
	then
		echo -n "create misc.img..."
		ln -s -f `echo $MISC_IMG | sed "s;$TOP_DIR;$RELATIVE_TOP_DIR;"` $ROCKDEV/misc.img
		echo "done."
	else
		echo "warning: $MISC_IMG not found!"
	fi
fi

if [ $RK_OEM_DIR ]
then
	if [ -d $OEM_DIR ]
	then
		$MKIMAGE $OEM_DIR $ROCKDEV/oem.img $RK_OEM_FS_TYPE
	else
		echo "warning: $OEM_DIR  not found!"
	fi
fi

if [ $RK_USERDATA_DIR ]
then
	if [ -d $USER_DATA_DIR ]
	then
		$MKIMAGE $USER_DATA_DIR $ROCKDEV/userdata.img $RK_USERDATA_FS_TYPE
	else
		echo "warning: $USER_DATA_DIR not found!"
	fi
fi

if [ -f $UBOOT_IMG ]
then
        echo -n "create uboot.img..."
        ln -s -f `echo $UBOOT_IMG | sed "s;$TOP_DIR;$RELATIVE_TOP_DIR;"` $ROCKDEV/uboot.img
        echo "done."
else
        echo -e "\e[31m error: $UBOOT_IMG not found! \e[0m"
fi

if [ -f $TRUST_IMG ]
then
        echo -n "create trust.img..."
        ln -s -f `echo $TRUST_IMG | sed "s;$TOP_DIR;$RELATIVE_TOP_DIR;"` $ROCKDEV/trust.img
        echo "done."
else
        echo -e "\e[31m error: $TRUST_IMG not found! \e[0m"
fi

if [ -f $LOADER ]
then
        echo -n "create loader..."
        ln -s -f `echo $LOADER | sed "s;$TOP_DIR;$RELATIVE_TOP_DIR;"` $ROCKDEV/MiniLoaderAll.bin
        echo "done."
else
	echo -e "\e[31m error: $LOADER not found,or there are multiple loaders! \e[0m"
	rm $LOADER
fi

#if [ -f $SPINOR_LOADER ]
#then
#        echo -n "create spinor loader..."
#        ln -s -f $SPINOR_LOADER $ROCKDEV/MiniLoaderAll_SpiNor.bin
#        echo "done."
#else
#	rm $SPINOR_LOADER_PATH 2>/dev/null
#fi

if [ $RK_BOOT_IMG ]
then
	if [ -f $BOOT_IMG ]
	then
		echo -n "create boot.img..."
		ln -s -f `echo $BOOT_IMG | sed "s;$TOP_DIR;$RELATIVE_TOP_DIR;"` $ROCKDEV/boot.img
		echo "done."
	else
		echo "warning: $BOOT_IMG not found!"
	fi
fi

if [ $RK_CFG_RAMBOOT ]
then
	if [ -f $RAMBOOT_IMG ]
	then
	        echo -n "create boot.img..."
	        ln -s -f `echo $RAMBOOT_IMG | sed "s;$TOP_DIR;$RELATIVE_TOP_DIR;"` $ROCKDEV/boot.img
	        echo "done."
	else
		echo "warning: $RAMBOOT_IMG not found!"
	fi
fi
echo -e "\e[36m Image: image in rockdev is ready \e[0m"
