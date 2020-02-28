/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#ifndef TENGINE_GRAPH_CONVOLUTION_HPP
#define TENGINE_GRAPH_CONVOLUTION_HPP

#define FLOAT_TO_REALSIZE (4)

#include "tengine_c_api.h"
#include "tengine_c_compat.h"
#include "tengine_operations.h"

/*!
 * @brief
 *
 * @param [in] graph:
 * @param [in] node_name:
 * @param [in] inch:
 * @param [in] in_h:
 * @param [in] in_w:

 * @return 0: success, -1: fail.
 */
int create_input_node(graph_t graph, const char* node_name, int inch, int in_h, int in_w);

/*!
 * @brief
 *
 * @param [in] graph:
 * @param [in] node_name:
 * @param [in] TODO

 * @return 0: success, -1: fail.
 */
int create_conv_node(graph_t graph, const char* node_name, const char* input_name, int in_h, int in_w, int out_h, int out_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int inch, int outch, int group,
    int dilation_h, int dilation_w, int activation, std::string padMode);

/*!
 * @brief
 *
 * @param [in] graph:
 * @param [in] node_name:
 * @param [in] TODO

 * @return 0: success, -1: fail.
 */
graph_t create_conv_graph(float *input_data, int inch, int group, int in_h, int in_w,
                        float *output_data, int outch, int out_h, int out_w,
                        int kernel_h, int kernel_w,
                        int stride_h,int stride_w,
                        int pad_h, int pad_w,  int dilation_h, int dilation_w, int activation,
                        float * teg_weight , float * teg_bias , std::string padMode);

#endif /* TENGINE_GRAPH_CONVOLUTION_HPP */
