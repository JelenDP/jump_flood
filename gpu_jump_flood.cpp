#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define CL_TARGET_OPENCL_VERSION 210

#include <vector>
#include <array>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>

/*#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif*/

#include <CL/cl2.hpp>

#include "define.h"

struct rawcolor { unsigned char r, g, b, a; };
struct rawcolor3{ unsigned char r, g, b; };
struct color    { float         r, g, b, a; };
struct Point    { cl_int3 point; };

int main()
{
    try
    {        
        int idx; //general index

        // Initialize vectors
        std::vector<cl_int3> map0(w*h);
        std::vector<cl_int3> zero(w*h);
        std::vector<cl_int3> seeds(n_seed);

        std::vector<color> seed_colors(n_seed);
        std::vector<color> colormap(w*h);
        std::vector<rawcolor> output_img(w*h);

        // Fill initialized vectors with random data
        std::mt19937 mersenne_engine{rnd_seed};  
        // generate seed colors
        std::uniform_real_distribution<float> dist{0, 1};

        auto gen_color = [&dist, &mersenne_engine](){     
                                                        color rgba;
                                                        rgba.r = dist(mersenne_engine);
                                                        rgba.g = dist(mersenne_engine);
                                                        rgba.b = dist(mersenne_engine);
                                                        rgba.a = 1.0f;
                                                        return rgba; 
                                                    };
        generate(seed_colors.begin(), seed_colors.end(), gen_color);

        /*seed_colors[0] = color{ 1.0f, 0.0f, 0.0f, 1.0f};
        seed_colors[1] = color{ 1.0f, 1.0f, 0.0f, 1.0f};
        seed_colors[2] = color{ 0.0f, 1.0f, 0.0f, 1.0f};
        seed_colors[3] = color{ 0.0f, 0.0f, 1.0f, 1.0f};*/ // colors for debug

        // generate seed points
        std::uniform_int_distribution<int> dist_w{0, w-1};
        std::uniform_int_distribution<int> dist_h{0, h-1};
        auto gen_seed= [&dist_w, &dist_h, &mersenne_engine](){
                                                    static int i;
                                                    cl_int3 point;
                                                    point.x = dist_w(mersenne_engine);
                                                    point.y = dist_h(mersenne_engine);
                                                    point.z = ++i;
                                                    return point;
                                                    };
        generate(seeds.begin(), seeds.end(), gen_seed);

        // fill the maps with seeds and colors
        auto gen_colormap = [](){return color{ 0.0f , 0.0f , 0.0f , 1.0f };};
        generate(colormap.begin(), colormap.end(), gen_colormap);

        auto gen_map = [](){return cl_int3{ 0 , 0 , 0 };};
        generate(map0.begin(), map0.end(), gen_map);
        generate(zero.begin(), zero.end(), gen_map);

        for (int i = 0; i < n_seed; i++){
            idx = ( seeds[i].y * w) + seeds[i].x;
            map0[idx].x = seeds[i].x ;
            map0[idx].y = seeds[i].y ;
            map0[idx].z = seeds[i].z ;

            colormap[idx].r = seed_colors[i].r;
            colormap[idx].g = seed_colors[i].g;
            colormap[idx].b = seed_colors[i].b;
            colormap[idx].a = seed_colors[i].a;
        }

        std::cout << "Seed random positions and colors were generated.\n";
        std::cout << " seed  x   y   R     G       B   \n";
        for (int i = 0; i < n_seed; i++){
            std::printf(" %3d  %3d %3d  %5.2f %5.2f %5.2f\n",seeds[i].z,seeds[i].x,seeds[i].y,
                                                            seed_colors[i].r*255.0f,seed_colors[i].g*255.0f,seed_colors[i].b*255.0f);
        }

        std::transform(colormap.cbegin(), colormap.cend(), output_img.begin(),
                [](color c){ return rawcolor{   (unsigned char)(c.r*255.0f),
                                                (unsigned char)(c.g*255.0f),
                                                (unsigned char)(c.b*255.0f),
                                                (unsigned char)(1.0f*255.0f) }; } );

        int res = stbi_write_png("../../results/res0.png", w, h, 4, output_img.data(), w*4);

        // OpenCL init:
        cl_int status = CL_SUCCESS;

        cl::CommandQueue queue = cl::CommandQueue::getDefault();

        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Platform platform{device.getInfo<CL_DEVICE_PLATFORM>()};

        std::cout << "Default queue on platform: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        std::cout << "Default queue on device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        auto  distance_op = "float r(int x0, int y0, int x1, int y1) { return (1.0f*(x0-x1)*(x0-x1)) + (1.0f*(y0-y1)*(y0-y1)); }";

        std::ifstream source_file{ "./../../jump_flood.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "./../../jump_flood.cl" };

        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} }.append(distance_op) };
        program.build({ device });

        auto jfa = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int>(program, "jump_flood");

        std::array<cl::Buffer, 2> buffer { cl::Buffer{ context, std::begin(map0), std::end(map0), false }, 
                                           cl::Buffer{ context, std::begin(zero), std::end(zero), false } };

        cl::copy(queue, std::begin(map0), std::end(map0), buffer[0]);
        cl::copy(queue, std::begin(zero), std::end(zero), buffer[1]);

        cl_int front = 0; //which buffer is read (front) and which is wroten (back)
        cl_int back  = 1;

        //log2(n) step
        std::cout << " Start Jump Flood algorithm \n";
        int ciklus = 1;
        for ( int step = w/2 ; step >= 1 ; step /= 2){
            std::cout << "  Step: " << ciklus << ", step length: " << step << "\n";

            cl::Event jfa_event{ (jfa)(cl::EnqueueArgs{queue,cl::NDRange{ (size_t)(w), (size_t)(h) } }, buffer[front], buffer[back], step)};
            jfa_event.wait();
            cl::copy(queue, buffer[back], std::begin(map0), std::end(map0));

            // fill the colormap with new seeds
            for (int x = 0; x < w; x++ ){
                for (int y = 0; y < h; y++ ){
                    idx = ( y * w ) + x ;
                    int seed = map0[idx].z - 1;
                    if (seed > n_seed) { throw std::runtime_error{ std::string{ "Invalid seed: " } + std::to_string(seed) }; }
                    if ( seed >= 0){
                        colormap[idx].r = seed_colors[seed].r;
                        colormap[idx].g = seed_colors[seed].g;
                        colormap[idx].b = seed_colors[seed].b;
                    }
                }
            }

            // save the results into an image
            std::transform(colormap.cbegin(), colormap.cend(), output_img.begin(),
            [](color c){ return rawcolor{   (unsigned char)(c.r*255.0f),
                                            (unsigned char)(c.g*255.0f),
                                            (unsigned char)(c.b*255.0f),
                                            (unsigned char)(1.0f*255.0f) }; } );

            std::string img_name = "../../results/res" + std::to_string(ciklus) + ".png";
            const char* img_name_c = img_name.c_str();
            res = stbi_write_png(img_name_c, w, h, 4, output_img.data(), w*4);
            
            // end of step
            if ( front == 0) { front = 1; back = 0; }else{ front = 0; back = 1;};
            ciklus++;

        }

        cl::finish();

        // set the original position with black
        for (int i = 0; i < n_seed; i++){
            idx = ( seeds[i].y * w) + seeds[i].x;
            colormap[idx].r = 0.0f;
            colormap[idx].g = 0.0f;
            colormap[idx].b = 0.0f;
            colormap[idx].a = 1.0f;
        }
        std::transform(colormap.cbegin(), colormap.cend(), output_img.begin(),
        [](color c){ return rawcolor{   (unsigned char)(c.r*255.0f),
                                        (unsigned char)(c.g*255.0f),
                                        (unsigned char)(c.b*255.0f),
                                        (unsigned char)(1.0f*255.0f) }; } );

        res = stbi_write_png("../../results/output.png", w, h, 4, output_img.data(), w*4);
        std::cout << " Jump Flood algorithm finished \n";


    }
    catch (cl::BuildError& error) // If kernel failed to build
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        for (const auto& log : error.getBuildLog())
        {
            std::cerr <<
                "\tBuild log for device: " <<
                log.first.getInfo<CL_DEVICE_NAME>() <<
                std::endl << std::endl <<
                log.second <<
                std::endl << std::endl;
        }

        std::exit(error.err());
    }
    catch (cl::Error& error) // If any OpenCL error occurs
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        std::exit(error.err());
    }
    catch (std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
	
	return 0;
}