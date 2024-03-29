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
struct color    { float         r, g, b, a; };
struct Point    { cl_int3 point; };
float r(int x0, int y0, int x1, int y1) { return (1.0f*(x0-x1)*(x0-x1)) + (1.0f*(y0-y1)*(y0-y1)); };

int main()
{
    try
    {        
        int idx; //general index
        const char spacer[60] = "---------------------------------------------";

        // Initialize vectors
        std::vector<cl_int3> map0(w*h);
        std::vector<cl_int3> zero(w*h);
        std::vector<cl_int3> out(w*h);
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

        int res = stbi_write_png("../../results/start.png", w, h, 4, output_img.data(), w*4);
        std::cout << spacer << std::endl;
        // CPU naiv Voronoi
        std::cout << " Start naiv Voronoi in CPU, ";
        auto start_step_cpu = std::chrono::high_resolution_clock::now();
        cl_int3 point = {w*w + h*h + 1, w*w + h*h + 1, 0};
        for (int x = 0; x < w; x++){
            for (int y = 0; y < h; y++){
                int idx = ( y * w ) + x;
                if (map0[idx].z == 0){
                    for ( int i = 0; i < n_seed; i++){
                        float d20 = r(point.x,      point.y,  x, y);
                        float d21 = r(seeds[i].x, seeds[i].y, x, y);
                         if (d20 > d21){
                            point.x = seeds[i].x;
                            point.y = seeds[i].y;
                            point.z = seeds[i].z;
                        }
                    }
                    out[idx].x = point.x;
                    out[idx].y = point.y;
                    out[idx].z = point.z;
                }
            }
        }
        auto end_step_cpu = std::chrono::high_resolution_clock::now();
        std::cout << " time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_step_cpu - start_step_cpu).count() << " ms \n";

        // save the results into an image
        for (int x = 0; x < w; x++ ){
            for (int y = 0; y < h; y++ ){
                idx = ( y * w ) + x ;
                int seed = out[idx].z - 1;
                if (seed > n_seed) { throw std::runtime_error{ std::string{ "Invalid seed: " } + std::to_string(seed) }; }
                if ( seed >= 0){
                    colormap[idx].r = seed_colors[seed].r;
                    colormap[idx].g = seed_colors[seed].g;
                    colormap[idx].b = seed_colors[seed].b;
                }
            }
        }
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
        res = stbi_write_png("../../results/naiv_cpu_output.png", w, h, 4, output_img.data(), w*4);
        
        std::cout << spacer << std::endl;
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

        // Naiv
        cl::Buffer buff0{ context, std::begin(map0), std::end(map0), false };
        cl::Buffer buff_seeds{ context, std::begin(seeds), std::end(seeds), true };

        cl::copy(queue, std::begin(map0), std::end(map0), buff0);
        cl::copy(queue, std::begin(seeds), std::end(seeds), buff_seeds);

        auto naiv = cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "naiv_voronoi");

        std::cout << " Start naiv Voronoi in GPU, ";
        auto start_step = std::chrono::high_resolution_clock::now();
        cl::Event naiv_event{ (naiv)(cl::EnqueueArgs{queue,cl::NDRange{ (size_t)(w), (size_t)(h) } }, buff0, buff_seeds)};
        naiv_event.wait();
        auto end_step = std::chrono::high_resolution_clock::now();
        std::cout << " time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_step - start_step).count() << " ms \n";
        cl::copy(queue, buff0, std::begin(map0), std::end(map0));
        cl::finish();

        // save the results into an image
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
        res = stbi_write_png("../../results/naiv_output.png", w, h, 4, output_img.data(), w*4);
    std::cout << spacer << std::endl;
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