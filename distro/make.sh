#!/bin/bash

set -e
DISTRO_DIR=$(dirname $(realpath "$0"))
TOP_DIR=$DISTRO_DIR/..

source $TOP_DIR/device/rockchip/.BoardConfig.mk
source $DISTRO_DIR/envsetup.sh
source $OUTPUT_DIR/.config
MIRROR_FILE=$OUTPUT_DIR/.mirror
DISTRO_CONFIG=$OUTPUT_DIR/.config
ROOTFS_DEBUG_EXT4=$IMAGE_DIR/rootfs.debug.ext4
ROOTFS_DEBUG_SQUASHFS=$IMAGE_DIR/rootfs.debug.squashfs
ROOTFS_EXT4=$IMAGE_DIR/rootfs.ext4
ROOTFS_SQUASHFS=$IMAGE_DIR/rootfs.squashfs
BUILD_PACKAGE=$1
SUITE=buster

if [ $SUITE==buster ] || [ $SUITE==stretch ] || [ $SUITE==sid ] || [ $SUITE==testing ];then
	OS=debian
elif [ $SUITE==bionic ] || [ $SUITE==xenial ] || [ $SUITE==trusty ];then
	OS=ubuntu
fi

log() {
    local format="$1"
    shift
    printf -- "$format\n" "$@" >&2
}

die() {
    local format="$1"
    shift
    log "E: $format" "$@"
    exit 1
}

run() {
    log "I: Running command: %s" "$*"
    "$@"
}

clean()
{
	rm -rf $OUTPUT_DIR
}

pack_squashfs()
{
	SRC=$1
	DST=$2
	mksquashfs $SRC $DST -noappend -comp gzip
}

pack_ext4()
{
	SRC=$1
	DST=$2
	SIZE=`du -sk --apparent-size $SRC | cut --fields=1`
	inode_counti=`find $SRC | wc -l`
	inode_counti=$[inode_counti+512]
	EXTRA_SIZE=$[inode_counti*4]
	SIZE=$[SIZE+EXTRA_SIZE]
	echo "genext2fs -b $SIZE -N $inode_counti -d $SRC $DST"
	genext2fs -b $SIZE -N $inode_counti -d $SRC $DST
	e2fsck -fy $DST
#	if [ -x $DISTRO_DIR/../device/rockchip/common/mke2img.sh ];then
#		$DISTRO_DIR/../device/rockchip/common/mke2img.sh $SRC $DST
#	fi
}

target_clean()
{
	system=$1
	for pkg in $(cat $DISTRO_DIR/configs/build.config)
	do
		if [ x$pkg != x`grep $pkg $DISTRO_CONFIG` ];then
			sudo chroot $system apt-get remove -y $pkg
		fi
	done

	sudo chroot $system apt-get autoclean -y
	sudo chroot $system apt-get clean -y
	sudo chroot $system apt-get autoremove -y
	sudo rm -rf $system/usr/share/locale/*
	sudo rm -rf $system/usr/share/man/*
	sudo rm -rf $system/usr/share/doc/*
	sudo rm -rf $system/usr/include/*
	sudo rm -rf $system/var/log/*
	sudo rm -rf $system/var/lib/apt/lists/*
	sudo rm -rf $system/var/cache/*
	echo "remove unused dri..."
	if [ $DISTRO_ARCH = arm64 ];then
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/msm_dri.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/nouveau_dri.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/nouveau_drv_video.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/nouveau_vieux_dri.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/r200_dri.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/r300_dri.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/r600_dri.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/r600_drv_video.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/radeon_dri.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/radeonsi_dri.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/radeonsi_drv_video.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/tegra_dri.so
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/dri/vc4_dri.so
	elif [ $DISTRO_ARCH = arm ];then
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/msm_dri.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/nouveau_dri.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/nouveau_drv_video.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/nouveau_vieux_dri.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/r200_dri.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/r300_dri.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/r600_dri.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/r600_drv_video.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/radeon_dri.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/radeonsi_dri.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/radeonsi_drv_video.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/tegra_dri.so
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/dri/vc4_dri.so
	fi
	echo "remove vdpau..."
	if [ $DISTRO_ARCH = arm64 ];then
		sudo rm -rf $system/usr/lib/aarch64-linux-gnu/vdpau
	elif [ $DISTRO_ARCH = arm ];then
		sudo rm -rf $system/usr/lib/arm-linux-gnueabihf/vdpau
	fi
	sudo rm -rf $system/sdk
}

pack()
{
	echo "packing rootfs image..."
#	rm -rf $ROOTFS_DIR
#	cp -ar $TARGET_DIR $ROOTFS_DIR
#	target_clean $ROOTFS_DIR
	if [ $RK_ROOTFS_TYPE = ext4 ];then
		pack_ext4 $TARGET_DIR $ROOTFS_EXT4
	elif [ $RK_ROOTFS_TYPE = squashfs ];then
		pack_squashfs $ROOTFS_DIR $ROOTFS_SQUASHFS
	fi
}

build_package()
{
	local pkg=$1
	echo "building package $pkg"
	if [ -x $PACKAGE_DIR/$pkg/make.sh ];then
		eval local dependence=`grep DEPENDENCIES $PACKAGE_DIR/$pkg/make.sh | cut -d = -f 2`
		echo "dependence=$dependence"
		if [ -n "$dependence" ];then
			for d in $dependence
			do
				build_package $d
			done
		fi
		run $PACKAGE_DIR/$pkg/make.sh
		echo "build $pkg done!!!"
	fi
}

build_packages()
{
	echo "building package: $RK_PKG"
	for p in $(ls $DISTRO_DIR/package/);do
	[ -d $DISTRO_DIR/package/$p ] || continue
	local config=BR2_PACKAGE_$(echo $p|tr 'a-z-' 'A-Z_')
	local build=$(eval echo -n \$$config)
	#echo "Build $pkg($config)? ${build:-n}"
	[ x$build == xy ] && build_package $p
	done
}

init()
{
	mkdir -p $OUTPUT_DIR $BUILD_DIR $TARGET_DIR $IMAGE_DIR $MOUNT_DIR $SYSROOT_DIR $TARGET_DIR/etc/apt/sources.list.d

	ARCH=$RK_ARCH
	if [ -z $ARCH ];then
		ARCH=arm64
	fi

	while read line1; do INSTALL_PKG="$INSTALL_PKG $line1"; done < "$OUTPUT_DIR/.install"
	while read line2; do SYSROOT_PKG="$SYSROOT_PKG $line2"; done < "$CONFIGS_DIR/rockchip/sysroot.install"
	#while read line3; do RK_PKG="$RK_PKG $line3"; done < "$CONFIGS_DIR/$RK_CONFIG"
        if [ ! -e $OUTPUT_DIR/.mirror ];then
		echo "find the fastest mirror"
		MIRROR=`$SCRIPTS_DIR/get_mirror.sh $OS $ARCH`
		echo $MIRROR > $OUTPUT_DIR/.mirror
	else
		MIRROR=`cat $OUTPUT_DIR/.mirror`
        fi
}

build_target_base()
{
if [ ! -e $OUTPUT_DIR/.targetpkg.done ];then
	echo "build target $OS $SUITE $ARCH package: $INSTALL_PKG"
	run $SCRIPTS_DIR/multistrap_build.sh -a $ARCH -b $SCRIPTS_DIR/debconfseed.txt -c $SCRIPTS_DIR/multistrap.conf -d $TARGET_DIR -m $MIRROR -p "$INSTALL_PKG" -s $SUITE
	$SCRIPTS_DIR/fix_link.sh $TARGET_DIR/usr/lib/$TOOLCHAIN
	#run $SCRIPTS_DIR/debootstrap_build.sh -a $ARCH -d $OUTPUT_DIR/debootstrap -m $MIRROR -p "$PACKAGES" -s $SUITE
	echo "deb [arch=$ARCH] $MIRROR $SUITE main" > $TARGET_DIR/etc/apt/sources.list.d/multistrap-debian.list
	touch $OUTPUT_DIR/.targetpkg.done
else
	echo "$OS $ARCH $SUITE package already installed for target, skip"
fi
}

build_sysroot()
{
if [ ! -e $OUTPUT_DIR/.sysrootpkg.done ];then
        echo "build sysroot package for $OS $SUITE: $SYSROOT_PKG"
        run $SCRIPTS_DIR/multistrap_build.sh -a $ARCH -b $SCRIPTS_DIR/debconfseed.txt -c $SCRIPTS_DIR/multistrap.conf -d $SYSROOT_DIR -m $MIRROR -p "$SYSROOT_PKG" -s $SUITE
	$SCRIPTS_DIR/fix_link.sh $SYSROOT_DIR/usr/lib/$TOOLCHAIN
        touch $OUTPUT_DIR/.sysrootpkg.done
else
        echo "$OS $ARCH $SUITE package already installed for sysroot, skip"
fi
}


build_all()
{
	init
	build_sysroot
	build_target_base
	build_packages
	run rsync -a --ignore-times --keep-dirlinks --chmod=u=rwX,go=rX --exclude .empty $OVERLAY_DIR/ $TARGET_DIR/
	pack
}

main()
{
	if [ x$1 == ximage ];then
		init
		pack
		exit 0
	elif [ x$1 == xsysroot ];then
		rm -f $OUTPUT_DIR/.sysrootpkg.done
		init
		build_sysroot
		exit 0
	elif [ x$1 == xtarget ];then
		rm -f $OUTPUT_DIR/.targetpkg.done
		init
		build_target_base
		exit 0
	elif [ -x $PACKAGE_DIR/$1/make.sh ];then
		ARCH=$RK_ARCH
		build_package $1
		exit 0
	else
		build_all
		exit 0
	fi
}

main "$@"
