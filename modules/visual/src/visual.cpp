// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {
namespace vs {

static bool readPointCloudOrMesh(const std::string& filename, OutputArray vertices, OutputArray triangles,
              OutputArray normals, OutputArray texturecoords, OutputArray attributes)
{
    FILE* f = fopen(filename.c_str(), "rb");
    if(!f) return false;
    bool ok = 
        isOBJ(filename, f) ? readOBJ(f, vertices, triangles, normals, texturecoords, attributes) :
        isPLY(filename, f) ? readPLY(f, vertices, triangles, normals, texturecoords, attributes) :
        false;
    fclose(f);
    return ok;
}

static bool writePointCloudOrMesh(const std::string& filename, InputArray vertices, InputArray triangles,
                                  InputArray normals, InputArray texturecoords, InputArray attributes)
{
    size_t len = filename.size();
    if(len > 4 && filename.substr(len-4, 4) == ".obj") {
        return writeOBJ(filename, vertices, triangles, normals, texturecoords, attributes);
    } else if(len > 4 && filename.substr(len-4, 4) == ".ply") {
        return writePLY(filename, vertices, triangles, normals, texturecoords, attributes);
    } else {
        return false;
    }
}
    
bool readPointCloud(const std::string& filename, OutputArray vertices, OutputArray normals,
                    OutputArray texturecoords, OutputArray attributes)
{
    return readPointCloudOrMesh(filename, vertices, noArray(), normals, texturecoords, attributes);
}

bool readMesh(const std::string& filename, OutputArray vertices, OutputArray triangles,
              OutputArray normals, OutputArray texturecoords, OutputArray attributes)
{
    return readPointCloudOrMesh(filename, vertices, triangles, normals, texturecoords, attributes);
}

bool writePointCloud(const std::string& filename, InputArray vertices, InputArray normals,
                    InputArray texturecoords, InputArray attributes)
{
    return writePointCloudOrMesh(filename, vertices, noArray(), normals, texturecoords, attributes);
}

bool writeMesh(const std::string& filename, InputArray vertices, InputArray triangles,
              InputArray normals, InputArray texturecoords, InputArray attributes)
{
    return writePointCloudOrMesh(filename, vertices, triangles, normals, texturecoords, attributes);
}
    
}}

/* End of file. */
