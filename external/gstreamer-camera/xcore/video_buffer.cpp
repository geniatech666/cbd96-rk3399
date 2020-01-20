/*
 * video_buffer.cpp - video buffer base
 *
 *  Copyright (c) 2014-2015 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "video_buffer.h"
#include <linux/videodev2.h>

namespace XCam {

VideoBufferPlanarInfo::VideoBufferPlanarInfo ()
{
    width = 0;
    height = 0;
    pixel_bytes = 0;
}

VideoBufferInfo::VideoBufferInfo ()
{
    format = 0;
    color_bits = 8;
    width = 0;
    height = 0;
    aligned_width = 0;
    aligned_height = 0;
    size = 0;
    components  = 0;
    xcam_mem_clear (strides);
    xcam_mem_clear (offsets);
}

bool
VideoBufferInfo::init (
    uint32_t format,
    uint32_t width, uint32_t height,
    uint32_t aligned_width, uint32_t aligned_height,
    uint32_t size)
{

    XCamVideoBufferInfo *info = this;

    return (xcam_video_buffer_info_reset (
                info, format, width, height, aligned_width, aligned_height, size) == XCAM_RETURN_NO_ERROR);
}

bool
VideoBufferInfo::get_planar_info (
    VideoBufferPlanarInfo &planar, const uint32_t index) const
{
    const XCamVideoBufferInfo *info = this;
    XCamVideoBufferPlanarInfo *planar_info = &planar;
    return (xcam_video_buffer_get_planar_info (info, planar_info, index) == XCAM_RETURN_NO_ERROR);
}

};
