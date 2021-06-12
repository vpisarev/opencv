/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_VISUAL_HPP
#define OPENCV_VISUAL_HPP

#include "opencv2/core.hpp"

/**
  @defgroup visual N-D data visualization
*/

//////////////////////////////// image codec ////////////////////////////////
namespace cv
{
namespace vs
{

//! @addtogroup visual
//! @{

////////////////// very approximate API ///////////////////////
    
// maybe init should be called automatically in namedWindows and other such functions
CV_EXPORTS_W void init();

// probably, mesh/ptcloud read&write functions should be moved to 3D module    
CV_EXPORTS_W bool readPointCloud(const std::string& filename,
                                 OutputArray vertices, OutputArray normals,
                                 OutputArray texturecoords, OutputArray attributes);
CV_EXPORTS_W bool readMesh(const std::string& filename,
                           OutputArray vertices, OutputArray triangles,
                           OutputArray normals, OutputArray texturecoords,
                           OutputArray attributes);
CV_EXPORTS_W bool writePointCloud(const std::string& filename,
                                  InputArray vertices, InputArray normals,
                                  InputArray texturecoords, InputArray attributes);
CV_EXPORTS_W bool writeMesh(const std::string& filename,
                            InputArray vertices, InputArray triangles,
                            InputArray normals, InputArray texturecoords,
                            InputArray attributes);

// maybe OpenGL should be exposed to the user?
// maybe there should be wrappers, like context, framebuffer etc.
// The current viz module can be taken as a base.
CV_EXPORTS_W void namedWindow(const std::string& wname);
CV_EXPORTS_W int waitKey(int delay=0);
CV_EXPORTS_W void imshow(const std::string& wname, InputArray image);
CV_EXPORTS_W void showPointCloud(const std::string& wname, InputArray vertices,
                                InputArray attributes, InputArray R, InputArray tvec);
CV_EXPORTS_W void showMesh(const std::string& wname, InputArray vertices, InputArray triangles,
                          InputArray attributes, InputArray R, InputArray tvec);

//! @} visual

}} // cv

#endif //OPENCV_VISUAL_HPP
