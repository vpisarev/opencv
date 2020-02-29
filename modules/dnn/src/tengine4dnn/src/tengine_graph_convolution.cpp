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

#include <iostream>
#include "../include/tengine_graph_convolution.hpp"

int create_input_node(graph_t graph, const char* node_name, int inch, int in_h, int in_w)
{
    node_t node     = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, inch, in_h, in_w};
    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_conv_node(graph_t graph, const char* node_name, const char* input_name, int in_h, int in_w, int out_h, int out_w, 
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int inch, int outch, int group, 
    int dilation_h, int dilation_w, int activation, std::string padMode)
{
    node_t conv_node      = create_graph_node(graph, node_name, "Convolution");
    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if (input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << std::endl;
        return -1;
    }

    set_node_input_tensor(conv_node, 0, input_tensor);
    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);

    set_node_output_tensor(conv_node, 0, output_tensor, TENSOR_TYPE_VAR);
    release_graph_tensor(output_tensor);

    /* weight */
    std::string weight_name(node_name);
    weight_name += "/weight";

    node_t w_node = create_graph_node(graph, weight_name.c_str(), "Const");
    tensor_t w_tensor = create_graph_tensor(graph, weight_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
    set_node_input_tensor(conv_node, 1, w_tensor);
    int w_dims[] = {outch, inch / group, kernel_h, kernel_w};

    set_tensor_shape(w_tensor, w_dims, 4);

    release_graph_node(w_node);
    release_graph_tensor(w_tensor);

    /* bias */
    std::string bias_name(node_name);
    bias_name += "/bias";

    node_t b_node = create_graph_node(graph, bias_name.c_str(), "Const");
    tensor_t b_tensor = create_graph_tensor(graph, bias_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);
    int b_dims[] = {outch};

    set_tensor_shape(b_tensor, b_dims, 1);

    set_node_input_tensor(conv_node, 2, b_tensor);
    release_graph_node(b_node);
    release_graph_tensor(b_tensor);

    int pad_h1 = pad_h;
    int pad_w1 = pad_w;

    if (!padMode.empty())
    {
        if (padMode == "SAME")
        {
            int out_h_temp = (in_h-kernel_h + 2*pad_h)/stride_h + 1;
            int out_w_temp = (in_w-kernel_w + 2*pad_w)/stride_w + 1;

            if (out_h_temp < out_h)
                pad_h1 += 1;
            if (out_w_temp < out_w)    
                pad_w1 += 1;
        }
    }

    /* attr */
    set_node_attr_int(conv_node, "kernel_h", &kernel_h);
    set_node_attr_int(conv_node, "kernel_w", &kernel_w);
    set_node_attr_int(conv_node, "stride_h", &stride_h);
    set_node_attr_int(conv_node, "stride_w", &stride_w);
    set_node_attr_int(conv_node, "pad_h0", &pad_h);
    set_node_attr_int(conv_node, "pad_w0", &pad_w);
    set_node_attr_int(conv_node, "pad_h1", &pad_h1);
    set_node_attr_int(conv_node, "pad_w1", &pad_w1);
    set_node_attr_int(conv_node, "output_channel", &outch);
    set_node_attr_int(conv_node, "group", &group);
    set_node_attr_int(conv_node, "dilation_h", &dilation_h);
    set_node_attr_int(conv_node, "dilation_w", &dilation_w);
    set_node_attr_int(conv_node, "activation", &activation); 

    release_graph_node(conv_node);

    return 0;
}

graph_t create_conv_graph(float *input_data, int inch, int group, int in_h, int in_w, 
                        float *output_data, int outch, int out_h, int out_w, 
                        int kernel_h, int kernel_w, 
                        int stride_h,int stride_w,
                        int pad_h, int pad_w,  int dilation_h, int dilation_w, int activation, 
                        float * teg_weight , float * teg_bias , std::string padMode)
{
    node_t    conv_node     = NULL;

    tensor_t  input_tensor  = NULL;
    tensor_t  output_tensor = NULL;
    tensor_t  weight_tensor = NULL;
    tensor_t  bias_tensor   = NULL;

    /* create graph for convolution */
    int in_size  = in_h * in_w * inch;
    int out_size  = out_h * out_w * outch;
    int weight_size = outch * (inch / group) * kernel_w * kernel_h;
    int bias_size = outch;

    int buf_size  = 0;
    int input_num = 0;

    /* create graph */
    graph_t graph = create_graph(nullptr, nullptr, nullptr);

    if(graph == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << std::endl;
    }

    const char* input_name = "data";
    const char* conv_name  = "conv";

    if (create_input_node(graph, input_name, inch, in_h, in_w) < 0)
    {
        std::cout << "create input failed\n";
    }

    if (create_conv_node(graph, conv_name, input_name, in_h, in_w, out_h, out_w, kernel_h, kernel_w, 
        stride_h, stride_w, pad_h, pad_w, inch, outch, group, dilation_h, dilation_w, activation, padMode) < 0)
    {
        std::cout << "create conv node failed\n";
    }

    /* set input/output node */
    const char* inputs_name[]  = {input_name};
    const char* outputs_name[] = {conv_name};

    if (set_graph_input_node(graph, inputs_name, sizeof(inputs_name) / sizeof(char*)) < 0)
    {
        std::cout << "set inputs failed: ERRNO: " << get_tengine_errno() << std::endl;
    }

    if (set_graph_output_node(graph, outputs_name, sizeof(outputs_name) / sizeof(char*)) < 0)
    {
        std::cout << "set outputs failed: ERRNO: " << get_tengine_errno() << std::endl;
    }

    /* set input data */
    input_tensor = get_graph_input_tensor(graph, 0, 0);
    buf_size     = get_tensor_buffer_size(input_tensor);
    if (buf_size != in_size * FLOAT_TO_REALSIZE)
    {
        printf("Input data size check failed . buf_size = %d , input_data_size = %d \n", buf_size, in_size);
    }

    set_tensor_buffer(input_tensor, (float *)input_data, buf_size);
    release_graph_tensor(input_tensor);

    /* create convolution node */
    /* set weight node */
    conv_node     = get_graph_node(graph, "conv");
    weight_tensor = get_node_input_tensor(conv_node, 1);
    buf_size      = get_tensor_buffer_size(weight_tensor);

    if (buf_size != weight_size * FLOAT_TO_REALSIZE)
    {
        printf("Input weight size check failed . buf_size = %d , input_weight_size = %d \n", buf_size, weight_size);
    }
    set_tensor_buffer(weight_tensor, teg_weight, buf_size);

    /* set bias node */
    input_num = get_node_input_number(conv_node);
    if (input_num > 2)
    {
        bias_tensor = get_node_input_tensor(conv_node, 2);
        buf_size    = get_tensor_buffer_size(bias_tensor);
        if (buf_size != bias_size * FLOAT_TO_REALSIZE)
        {
            printf("Input bias size check failed . buf_size = %d , bias_size = %d \n", buf_size, bias_size);
            // return;
        }
        set_tensor_buffer(bias_tensor, teg_bias, buf_size);
    }

    /* set output data */
    output_tensor = get_node_output_tensor(conv_node, 0);
    int ret = set_tensor_buffer(output_tensor, output_data, out_size * FLOAT_TO_REALSIZE);
    if(ret)
    {
        printf("Set output tensor buffer failed . ret = %d \n", ret);
        // return;
    }

    return graph;
}

void tengine_set_Winograd_off(bool flag)
{
    // Default not using the winograd algorithm.
    setenv("NO_WINO","1",flag);
}

bool tengine_forward(float *input_, int inch, int group, int in_h, int in_w, 
                        float *output_, int out_b, int outch, int out_h, int out_w, 
                        float *kernel_, int kernel_s ,int kernel_h, int kernel_w, 
                        float *teg_bias, int stride_h,int stride_w,
                        int pad_h, int pad_w,  int dilation_h, int dilation_w, 
                        size_t wstep, std::string padMode)
{
    graph_t graph = NULL;
    float *teg_weight = NULL;
    int kernel_inwh = (inch / group) * kernel_w * kernel_h;
    // Do not using the activation fuse mode, just convolution only.
    int activation = -1; 
  
    if (kernel_s == 2 && kernel_h == kernel_w && pad_h == pad_w
                                    && dilation_h == dilation_w && stride_h == stride_w
                                    && out_b == 1 && pad_h < 10) // just for Conv2D
    {

        tengine_set_Winograd_off(true);
    
        // weight
        if (kernel_inwh != wstep)
        {
            teg_weight = (float *)malloc(kernel_inwh * outch * sizeof(float));
            for (int i=0; i<outch; i++)
            {
                memcpy(teg_weight+i*kernel_inwh, kernel_+i*wstep, kernel_inwh*FLOAT_TO_REALSIZE);
            }
        }
        else
        {
            teg_weight = kernel_;
        }

        /* initial the resoruce of tengine */
        init_tengine();

        /* create the convolution graph */
        graph = create_conv_graph( input_, inch, group, in_h, in_w,
                                    output_, outch, out_h, out_w,
                                    kernel_h, kernel_w, stride_h,stride_w,
                                    pad_h, pad_w, dilation_h, dilation_w, activation,
                                    teg_weight , teg_bias , padMode);

        /* prerun */
        if(prerun_graph(graph) < 0)
        {
            std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
            return false ;
        }

        /* run */
        if(run_graph(graph, 1) < 0)
        {
            std::cerr << "run_graph failed \n";
            return false ;
        }

        /* release the resource */
        // if(teg_weight)
        //     free(teg_weight);
        postrun_graph(graph);
        destroy_graph(graph);
    }
    else     
    {
 //       printf("Not Support by tengine .\n");
        return false ;
    }
    return true ;
}
