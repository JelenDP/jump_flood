#include "./../../define.h"


float r(int x0, int y0, int x1, int y1);

//------------------------------------------------------------------------------------------------------------
kernel void naiv_voronoi( global int3* buff0, //main buffer
                          global int3* seeds  //seeds
                        )
{
	int x = get_global_id(0); // current x
    int y = get_global_id(1); // current y
    
    int idx0 = ( y * w ) + x;   //idx of thread
    int3 point = (int3)  {w*w + h*h + 1, w*w + h*h + 1, 0};      

    // all thread read it's position if it's has a seed then copy it to point else point is zero point
    if (buff0[idx0].z == 0){
        for ( int i = 0; i < n_seed; i++){
            float d20 = r(point.x,      point.y,  x, y);
            float d21 = r(seeds[i].x, seeds[i].y, x, y);
            if (d20 > d21){
                point.x = seeds[i].x;
                point.y = seeds[i].y;
                point.z = seeds[i].z;
            }
        }
        buff0[idx0].x = point.x;
        buff0[idx0].y = point.y;
        buff0[idx0].z = point.z;
    }
}

//------------------------------------------------------------------------------------------------------------
kernel void jump_flood( global int3* buff0, //main buffer
                        global int3* buff1, //copy buffer
                        int step            //step lenght
                    )
{
	int x = get_global_id(0); // current x
    int y = get_global_id(1); // current y
    
    int idx0 = ( y * w ) + x;   //idx of thread
    int3 point = (int3)  {w*w + h*h + 1, w*w + h*h + 1, 0};      

    const int2 direction[8] = { {  1,  0 } ,  // <
                                { -1,  0 } ,  // >
                                {  0,  1 } ,  // v 
                                {  0, -1 } ,  // ^
                                {  1,  1 } ,  
                                { -1, -1 } ,
                                {  1, -1 } ,
                                { -1,  1 } } ;

    
    // all thread read it's position if it's has a seed then copy it to point else point is zero point
    if (buff0[idx0].z != 0){
        point.x = buff0[idx0].x;
        point.y = buff0[idx0].y;
        point.z = buff0[idx0].z;
    }

    // all thread read their 8 neighbours and if find a closer one then current copy it to point
    for (int i = 0; i < 8; i++){
        int yc =  y + step * direction[i].y;
        int xc =  x + step * direction[i].x;
        int idx = ( yc * w ) + xc;
        if (yc >= 0 && yc < h && xc >= 0 && xc < w) {
            if (buff0[idx].z != 0){
                float d20 = r(point.x,      point.y,      x, y);
                float d21 = r(buff0[idx].x, buff0[idx].y, x, y);
                if (d20 > d21){ 
                    point.x = buff0[idx].x;
                    point.y = buff0[idx].y;
                    point.z = buff0[idx].z;
                }
            }
        }
    }
    
    // write point to back buffer if it has seed
    if (point.z != 0){
        buff1[idx0].x = point.x;
        buff1[idx0].y = point.y;
        buff1[idx0].z = point.z;
    }
}
//------------------------------------------------------------------------------------------------------------
kernel void jump_flood_improved( global int3* buff_g, //main buffer
                                 local int3*  buff_l   //copy buffer
                                )
{
	int gx = get_global_id(0); // global x
    int gy = get_global_id(1); // global y

    int lx = get_local_id(0); // local x
    int ly = get_local_id(1); // local y

    const size_t wid = get_group_id(0);

    int steps = w / BS;
    int3 point = (int3) { 2*BS*BS + 1, 2*BS*BS + 1,  0};

    int idx0 = ly * BS + lx;   //idx of thread

    const int2 direction[8] = { {  1,  0 } ,  // <
                                { -1,  0 } ,  // >
                                {  0,  1 } ,  // v 
                                {  0, -1 } ,  // ^
                                {  1,  1 } ,  
                                { -1, -1 } ,
                                {  1, -1 } ,
                                { -1,  1 } } ;

    buff_l[idx0] = buff_g[ gy * w + wid * BS + lx];  // copy from global to local
    barrier(CLK_LOCAL_MEM_FENCE);                    // wait until copy done

    /*event_t read;
    async_work_group_copy(
        buff_l,
        buff_g + wid * BS * BS,
        BS * BS,
        read);
    wait_group_events(1, &read);
    barrier(CLK_LOCAL_MEM_FENCE);*/

    for (int step = BS/2 ; step >= 1 ; step /= 2){    // jump flood steps
        if (buff_l[idx0].z != 0){
            point = (int3) { buff_l[idx0].x, buff_l[idx0].y, buff_l[idx0].z };
        }
        for (int i = 0; i < 8; i++){
            int yc =  ly + step * direction[i].y;
            int xc =  lx + step * direction[i].x;
            int idx = ( yc * BS ) + xc;
            if (yc >= 0 && yc < BS && xc >= 0 && xc < BS) {
                if (buff_l[idx].z != 0){
                    float d20 = r(point.x,      point.y,        gx, gy);
                    float d21 = r(buff_l[idx].x, buff_l[idx].y, gx, gy);
                    if (d20 > d21){ 
                        point.x = buff_l[idx].x;
                        point.y = buff_l[idx].y;
                        point.z = buff_l[idx].z;
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // wait until all thread visit their neighobours
        buff_l[idx0].x = point.x;
        buff_l[idx0].y = point.y;
        buff_l[idx0].z = point.z; //block_i+1;//point.z;
    }

    /*event_t write;
    async_work_group_copy(
        buff_g + wid * BS * BS,
        buff_l,
        BS * BS,
        read);
    wait_group_events(1, &read);
    barrier(CLK_LOCAL_MEM_FENCE);*/

    buff_g[ gy * w + wid * BS + lx] = buff_l[idx0];  // copy from global to local
    barrier(CLK_LOCAL_MEM_FENCE);                    // wait until copy done
}

