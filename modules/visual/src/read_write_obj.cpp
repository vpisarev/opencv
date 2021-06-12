// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {
namespace vs {
    
bool isOBJ(const std::string&, FILE*) { return false; }
bool readOBJ(FILE*, OutputArray, OutputArray,
             OutputArray, OutputArray, OutputArray)
{
    return false;
}

bool writeOBJ(const std::string&, InputArray, InputArray,
              InputArray, InputArray, InputArray)
{
    return false;
}
    
}}

/* End of file. */
