// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include "opencv2/visual.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"

namespace cv { namespace vs {

bool isOBJ(const std::string& filename, FILE* f);
bool readOBJ(FILE* f, OutputArray vertices, OutputArray triangles,
             OutputArray normals, OutputArray texturecoords, OutputArray attributes);
bool writeOBJ(const std::string& filename, InputArray vertices, InputArray triangles,
              InputArray normals, InputArray texturecoords, InputArray attributes);
bool isPLY(const std::string& filename, FILE* f);
bool readPLY(FILE* f, OutputArray vertices, OutputArray triangles,
             OutputArray normals, OutputArray texturecoords, OutputArray attributes);
bool writePLY(const std::string& filename, InputArray vertices, InputArray triangles,
              InputArray normals, InputArray texturecoords, InputArray attributes);

}}

#endif
