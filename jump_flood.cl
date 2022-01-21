#include "./../../define.h"


float r(int x0, int y0, int x1, int y1);

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
