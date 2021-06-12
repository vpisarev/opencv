// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {
namespace vs {
    
bool isPLY(const std::string&, FILE*) { return false; }
bool readPLY(FILE*, OutputArray, OutputArray,
             OutputArray, OutputArray, OutputArray)
{
    return false;
}

bool writePLY(const std::string&, InputArray, InputArray,
              InputArray, InputArray, InputArray)
{
    return false;
}
    
}}

/* End of file. */
