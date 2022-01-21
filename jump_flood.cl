#include "./../../define.h"


float distance(int x0, int y0, int x1, int y1);
//float r(int x0, int y0, int x1, int y1) { return ((x0-x1)*(x0-x1)) + ((y0-y1)*(y0-y1)); }

kernel void jump_flood( global int3* buff0, //main buffer
                        global int3* buff1, //copy buffer
                        int step            //step lenght
                    )
{
	int x = get_global_id(0);
    int y = get_global_id(1);
    
    int idx0 = (y*w) + x;   //idx of thread
    int idx;                //idx of current neighbour
    int3 point; //(x,y,seed) of current point 
    float d20;              //distance from current seed
    float d21;              //distance from neighbour seed

    const int2 direction[8] = { {  1,  0 } ,  // <
                                { -1,  0 } ,  // >
                                {  0,  1 } ,  // v 
                                {  0, -1 } ,  // ^
                                {  1,  1 } ,  
                                { -1, -1 } ,
                                {  1, -1 } ,
                                { -1,  1 } } ;

    // all thread copy one point to its private memory if it's a seed point
    if (buff0[idx0].z != 0){
        point.x = buff0[idx0].x;
        point.y = buff0[idx0].y;
        point.z = buff0[idx0].z;
    
        // all thread which has seed write their seed to their 8 neighbour and to themself
        buff1[idx0].x = point.x;
        buff1[idx0].y = point.y;
        buff1[idx0].z = point.z;

        for (int i=0; i < 8; i++){
            idx = ( ( y + (step * direction[i].y) ) * w) + ( x + ( step * direction[i].x ) );
            if (idx >= 0 && idx <= w*h) {
                if (buff1[idx].z == 0){ // if it's an empty point copy current seed to it
                    buff1[idx].x = point.x;
                    buff1[idx].y = point.y;
                    buff1[idx].z = point.z;
                }else{ // if it already has seed, then calculate the distance
                    d20 = distance(point.x,      point.y,      x, y);
                    d21 = distance(buff1[idx].x, buff1[idx].y, x, y);
                    if (d20 < d21){ // if previous seed is farther than current seed, then copy current one
                        buff1[idx].x = point.x;
                        buff1[idx].y = point.y;
                        buff1[idx].z = point.z;
                    }
                }
            }
        }
    }
}
