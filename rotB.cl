#define SCALE %(scale)f
#define BLOCK_SIZE %(block_size)d
#define NL (%(block_size)d+2)
#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
#define MX %(mx)d
#define MY %(my)d
#define MZ %(mz)d
    
#define mm(i,j) (i>j)?i*(i+1)/2+j:j*(j+1)/2+i //"smart" index for triagngular matrices
    
__constant float ai[21] = {1./5,
                           3./40, 9./40,
                           44./45, -56./15, 32./9,
                           19372./6561, -25360./2187, 64448./6561, -212./729,
                           9017./3168, -355./33, 46732./5247, 49./176, -5103./18656,
                           35./384, 0., 500./1113, 125./192, -2187./6784, 11./84};    

__constant float bi[7] = {5179./57600, 0., 7571./16695, 393./640, -92097./339200, 187./2100};

__constant unsigned int ng[3] = {NX,NY,NZ};
__constant unsigned int bg[3] = {NY*NZ,NZ,1};
__constant unsigned int bl[3] = {NL*NL,NL,1};


struct rb_str {
    float3 force;
    float3 current;
};


float3 Cross(float3 U, float3 V)
{
    float3 Y;
    Y.x = U.y*V.z - U.z*V.y;
    Y.y = U.z*V.x - U.x*V.z;
    Y.z = U.x*V.y - U.y*V.x;
    
    return Y;
}



float3 Deriv(__local float3* Xl, unsigned int ldx, uchar dim, uchar order)
{  
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
       
    float3 out;
      
    if ((ii[dim] != 0) && (ii[dim] != (ng[dim]-1))) { 
        if (order == 1) out = Xl[ldx+bl[dim]]*(float3)(0.5) - Xl[ldx-bl[dim]]*(float3)(0.5);
        if (order == 2) out = Xl[ldx+bl[dim]] + Xl[ldx-bl[dim]] - (float3)(2)*Xl[ldx];              
    } else if (ii[dim] == 0) {
        if (order == 1) out = -(float3)(1.5)*Xl[ldx] + (float3)(2)*Xl[ldx+bl[dim]] - (float3)(0.5)*Xl[ldx+2*bl[dim]]; 
        if (order == 2) out = (float3)(2)*Xl[ldx] - (float3)(5)*Xl[ldx+bl[dim]] + (float3)(4)*Xl[ldx+2*bl[dim]] - Xl[ldx+3*bl[dim]];
    } else if (ii[dim] == (ng[dim]-1)) {
        if (order == 1) out = (float3)(1.5)*Xl[ldx] - (float3)(2)*Xl[ldx-bl[dim]] + (float3)(0.5)*Xl[ldx-2*bl[dim]];
        if (order == 2) out = (float3)(2)*Xl[ldx] - (float3)(5)*Xl[ldx-bl[dim]] + (float3)(4)*Xl[ldx-2*bl[dim]] - Xl[ldx-3*bl[dim]];
    }
    return out;
}



__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)))
void Jacobian(__global float3* X, __global float3* J)
{
    __local float3 Xl[NL*NL*NL];
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    unsigned int ll[3] = {get_local_id(0)+1,get_local_id(1)+1,get_local_id(2)+1};
    unsigned int ldx = ll[0]*bl[0]+ll[1]*bl[1]+ll[2]*bl[2];
    
    uchar i;
    
    Xl[ldx] = X[idx];
    
    for (i = 0; i < 3; i++) {
        if ((ll[i] == 1) && (ii[i] != 0)) Xl[ldx-bl[i]] = X[idx-bg[i]];
        if ((ll[i] == NL-2) && (ii[i] != ng[i]-1)) Xl[ldx+bl[i]] = X[idx+bg[i]];
        barrier(CLK_LOCAL_MEM_FENCE);
        J[idx + i*NX*NY*NZ] = Deriv(Xl, ldx, i, 1);
    }
}



struct rb_str rotB(__local float3* Xl, __global float3* Bg, __global float3* DB)
{
    __local float3 DXl[NL*NL*NL];
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    unsigned int ll[3] = {get_local_id(0)+1,get_local_id(1)+1,get_local_id(2)+1};
    unsigned int ldx = ll[0]*bl[0]+ll[1]*bl[1]+ll[2]*bl[2];
    
    float3 dX[3],dXb[3],d2X[6],B,Bx,dB[3],dBx[3];
    float det_dX,d_det_dX[3];
    uchar i,j;
    
    //loading field to private memory
    B = Bg[idx];

    for (i = 0; i < 3; i++) {
        dB[i] = DB[idx + i*NX*NY*NZ];
        
        //computing first and second derivatives of X
        d2X[i*(i+3)/2] = Deriv(Xl, ldx, i, 2);
        dX[i] = Deriv(Xl, ldx, i, 1);
        
        //loading Jacobian to local memory
        if (i != 0) DXl[ldx] = dX[i];
        
        //computing cross-dimentional secong derivatives of X
        for (j = 0; j < i; j++) {
            if (ll[j] == 1) DXl[ldx-bl[j]] = Deriv(Xl, ldx-bl[j], i, 1);
            if (ll[j] == NL-2) DXl[ldx+bl[j]] = Deriv(Xl, ldx+bl[j], i, 1);
            barrier(CLK_LOCAL_MEM_FENCE);
            d2X[i*(i+1)/2+j] = Deriv(DXl, ldx, j, 1);
        }
    }

    //-----------------------------the rest of code uses only private memory-------------------------------------
    
    det_dX = dX[0].x*dX[1].y*dX[2].z + dX[0].y*dX[1].z*dX[2].x + dX[0].z*dX[1].x*dX[2].y - 
           dX[0].x*dX[1].z*dX[2].y - dX[0].y*dX[1].x*dX[2].z - dX[0].z*dX[1].y*dX[2].x;
    
    dXb[0] = Cross(dX[1], dX[2])/(float3)(det_dX);
    dXb[1] = Cross(dX[2], dX[0])/(float3)(det_dX);
    dXb[2] = Cross(dX[0], dX[1])/(float3)(det_dX);
    
    Bx.x = dX[0].x*B.x + dX[1].x*B.y + dX[2].x*B.z;
    Bx.y = dX[0].y*B.x + dX[1].y*B.y + dX[2].y*B.z;
    Bx.z = dX[0].z*B.x + dX[1].z*B.y + dX[2].z*B.z;
    
    Bx /= (float3)det_dX;
    
    for (i = 0; i < 3; i++) {
        d_det_dX[i] = 0.;   
        for (j = 0; j < 3; j++) d_det_dX[i] += d2X[mm(i,j)].x*dXb[j].x + d2X[mm(i,j)].y*dXb[j].y + d2X[mm(i,j)].z*dXb[j].z;
        d_det_dX[i] *= det_dX;
    }
       
    for (i = 0; i < 3; i++) {
        dBx[i] = -(float3)(d_det_dX[i])*Bx;

        dBx[i].x += dX[0].x*dB[i].x + dX[1].x*dB[i].y + dX[2].x*dB[i].z;
        dBx[i].y += dX[0].y*dB[i].x + dX[1].y*dB[i].y + dX[2].y*dB[i].z;
        dBx[i].z += dX[0].z*dB[i].x + dX[1].z*dB[i].y + dX[2].z*dB[i].z;
        
        dBx[i].x += d2X[mm(i,0)].x*B.x + d2X[mm(i,1)].x*B.y + d2X[mm(i,2)].x*B.z;
        dBx[i].y += d2X[mm(i,0)].y*B.x + d2X[mm(i,1)].y*B.y + d2X[mm(i,2)].y*B.z;
        dBx[i].z += d2X[mm(i,0)].z*B.x + d2X[mm(i,1)].z*B.y + d2X[mm(i,2)].z*B.z;

        dBx[i] /= (float3)(det_dX);
    }
    
    struct rb_str out;
    out.current = (float3)0;
    for (i = 0; i < 3; i++) {
        out.current += Cross(dXb[i], dBx[i]);
    }
    out.force = Cross(out.current, Bx);
    
    return out;
}



__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)))
void Dopr(__global float3* X, __global float3* X1, __global float3* B, __global float3* DB, float step, __global float* Error, __global float3* Current)
{       
    __local float3 Xl[NL*NL*NL];
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    unsigned int ll[3] = {get_local_id(0)+1,get_local_id(1)+1,get_local_id(2)+1};
    unsigned int ldx = ll[0]*bl[0]+ll[1]*bl[1]+ll[2]*bl[2];

    uchar i,j;
    
    bool not_border = true;
    for (i = 0; i < 3; i++) not_border = not_border && (ii[i] != 0) && (ii[i] != ng[i]-1);
    //not_border = (ii[2] != 0) && (ii[2] != ng[2]-1);
    
    Xl[ldx] = X[idx];
    for (i = 0; i < 3; i++) {
        if ((ll[i] == 2) && (ii[i] != 1)) Xl[ldx-2*bl[i]] = X[idx-2*bg[i]];
        if ((ll[i] == NL-3) && (ii[i] != ng[i]-2)) Xl[ldx+2*bl[i]] = X[idx+2*bg[i]];
        for (j = 0; j < i; j++) {
            if ((ll[i] == 1) && (ll[j] == 1) && (ii[i] != 0) && (ii[j] != 0)) Xl[ldx-bl[i]-bl[j]] = X[idx-bg[i]-bg[j]];
            if ((ll[i] == 1) && (ll[j] == NL-2) && (ii[i] != 0) && (ii[j] != ng[j]-1)) Xl[ldx-bl[i]+bl[j]] = X[idx-bg[i]+bg[j]];
            if ((ll[i] == NL-2) && (ll[j] == 1) && (ii[i] != ng[i]-1) && (ii[j] != 0)) Xl[ldx+bl[i]-bl[j]] = X[idx+bg[i]-bg[j]];
            if ((ll[i] == NL-2) && (ll[j] == NL-2) && (ii[i] != ng[i]-1) && (ii[j] != ng[j]-1)) Xl[ldx+bl[i]+bl[j]] = X[idx+bg[i]+bg[j]];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float3 Xp = X[idx];
    float3 ki[8];
    ki[6] = Xp;
    
    struct rb_str temp;
    
    for (i = 0; i < 6; i++) {     
        temp = rotB(Xl, B, DB);
        ki[i] = temp.force*step;
        if (not_border) ki[6] += ki[i]*bi[i];
        
        ki[7] = Xp;
        for (j = 0; j <= i; j++) 
            if (not_border) ki[7] += ai[mm(i,j)]*ki[j];
        Xl[ldx] = ki[7];
        barrier(CLK_LOCAL_MEM_FENCE);  
    } 
    
    X1[idx] = ki[7];
    Error[idx] = (pow(ki[6].x-ki[7].x,2.f)+pow(ki[6].y-ki[7].y,2.f)+pow(ki[6].z-ki[7].z,2.f))/(SCALE*SCALE);
    Current[idx] = temp.current;
}

