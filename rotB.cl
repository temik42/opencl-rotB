#define SCALE %(scale)f
#define NL %(block_size)d
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


float3 Cross(float3 U, float3 V)
{
    float3 Y;
    Y.x = U.y*V.z - U.z*V.y;
    Y.y = U.z*V.x - U.x*V.z;
    Y.z = U.x*V.y - U.y*V.x;
    
    return Y;
}



float3 Deriv(__local float3* Xl, uchar dim, uchar order)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
 
    unsigned int lx = get_local_id(0)+1;
    unsigned int ly = get_local_id(1)+1;
    unsigned int lz = get_local_id(2)+1;
    
    unsigned int nl = NL+2;
    
    unsigned int ldx = lz+nl*ly+nl*nl*lx;

    float3 out;
    
    unsigned int ii, ng, ll, bl;
    if (dim == 0) {ng = NX; ii = ix; ll = lx; bl = nl*nl;}
    if (dim == 1) {ng = NY; ii = iy; ll = ly; bl = nl   ;}
    if (dim == 2) {ng = NZ; ii = iz; ll = lz; bl = 1    ;}
      
    if ((ii != 0) && (ii != (ng-1))) { 
        if (order == 1) out = Xl[ldx+bl]*(float3)(0.5) - Xl[ldx-bl]*(float3)(0.5);
        if (order == 2) out = Xl[ldx+bl] + Xl[ldx-bl] - (float3)(2)*Xl[ldx];              
    } else if (ii == 0) {
        if (order == 1) out = -(float3)(1.5)*Xl[ldx] + (float3)(2)*Xl[ldx+bl] - (float3)(0.5)*Xl[ldx+2*bl]; 
        if (order == 2) out = (float3)(2)*Xl[ldx] - (float3)(5)*Xl[ldx+bl] + (float3)(4)*Xl[ldx+2*bl] - Xl[ldx+3*bl];
    } else if (ii == (ng-1)) {
        if (order == 1) out = (float3)(1.5)*Xl[ldx] - (float3)(2)*Xl[ldx-bl] + (float3)(0.5)*Xl[ldx-2*bl];
        if (order == 2) out = (float3)(2)*Xl[ldx] - (float3)(5)*Xl[ldx-bl] + (float3)(4)*Xl[ldx-2*bl] - Xl[ldx-3*bl];           
    }
    return out;
}





__kernel __attribute__((reqd_work_group_size(NL,NL,NL)))
void Jacobian(__global float3* X, __global float3* J)
{
    __local float3 Xl[(NL+2)*(NL+2)*(NL+2)];
    
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int idx = iz+NZ*iy+NY*NZ*ix;
 
    unsigned int lx = get_local_id(0)+1;
    unsigned int ly = get_local_id(1)+1;
    unsigned int lz = get_local_id(2)+1;
    
    unsigned int nl = NL+2;
    
    unsigned int ldx = lz+nl*ly+nl*nl*lx;
    
    unsigned int i;
    
    Xl[ldx] = X[idx];
    if ((lx == 1) && (ix != 0)) Xl[ldx-nl*nl] = X[idx-NY*NZ];
    if ((lx == nl-2) && (ix != NX-1)) Xl[ldx+nl*nl] = X[idx+NY*NZ];
    if ((ly == 1) && (iy != 0)) Xl[ldx-nl] = X[idx-NZ];
    if ((ly == nl-2) && (iy != NY-1)) Xl[ldx+nl] = X[idx+NZ];
    if ((lz == 1) && (iz != 0)) Xl[ldx-1] = X[idx-1];
    if ((lz == nl-2) && (iz != NZ-1)) Xl[ldx+1] = X[idx+1];
    barrier(CLK_LOCAL_MEM_FENCE); 
    
    for (i = 0; i < 3; i++) J[idx + i*NX*NY*NZ] = Deriv(Xl, i, 1);
}




//__kernel __attribute__((reqd_work_group_size(NL,NL,NL)))
float3 force(__local float3* Xl, __global float3* DX, __global float3* Bg, __global float3* DB)
{
    __local float3 DXl[(NL+2)*(NL+2)*(NL+2)];
    
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int idx = iz+NZ*iy+NY*NZ*ix;
 
    unsigned int lx = get_local_id(0)+1;
    unsigned int ly = get_local_id(1)+1;
    unsigned int lz = get_local_id(2)+1;
    
    unsigned int nl = NL+2;
    
    unsigned int ldx = lz+nl*ly+nl*nl*lx;
    
    float3 dX[3],dXb[3],d2X[6],B,Bx,dB[3],dBx[3];
    float det_dX,d_det_dX[3];
    uchar i,j;
    
    //loading field to private memory
    B = Bg[idx];
    
    //computing first and second derivatives of X
    for (i = 0; i < 3; i++) {
        dB[i] = DB[idx + i*NX*NY*NZ];
        dX[i] = Deriv(Xl, i, 1);
        d2X[i*(i+3)/2] = Deriv(Xl, i, 2);
        DX[idx + i*NX*NY*NZ] = dX[i];
    }
    //barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (i = 0; i < 3; i++) {
        
        
        //computing first and second derivatives of X
        //dX[i] = Deriv(Xl, i, 1);
        //d2X[i*(i+3)/2] = Deriv(Xl, i, 2);
        //DX[idx + i*NX*NY*NZ] = dX[i];
        
        //loading Jacobian to local memory
        if (i != 0) {
            DXl[ldx] = dX[i];
            if ((lx == 1) && (ix != 0)) DXl[ldx-nl*nl] = DX[idx-NY*NZ + i*NX*NY*NZ];
            if ((lx == nl-2) && (ix != NX-1)) DXl[ldx+nl*nl] = DX[idx+NY*NZ + i*NX*NY*NZ];
            if ((ly == 1) && (iy != 0)) DXl[ldx-nl] = DX[idx-NZ + i*NX*NY*NZ];
            if ((ly == nl-2) && (iy != NY-1)) DXl[ldx+nl] = DX[idx+NZ + i*NX*NY*NZ];
            if ((lz == 1) && (iz != 0)) DXl[ldx-1] = DX[idx-1 + i*NX*NY*NZ];
            if ((lz == nl-2) && (iz != NZ-1)) DXl[ldx+1] = DX[idx+1 + i*NX*NY*NZ];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        //computing cross-dimentional secong derivatives of X
        //the trick is that border points are processed using the derivatives calculated on the previous iteration
        for (j = 0; j < i; j++) d2X[i*(i+1)/2+j] = Deriv(DXl, j, 1);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //the rest of code uses only private memory
    
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
    
    float3 Ix = (float3)0;
    for (i = 0; i < 3; i++) {
        Ix += Cross(dXb[i], dBx[i]);
    }
    return Cross(Ix, Bx);
}



__kernel __attribute__((reqd_work_group_size(NL,NL,NL)))
void Dopr(__global float3* X, __global float3* DX, __global float3* B, __global float3* DB, float step, __global float* Error)
{       
    __local float3 Xl[(NL+2)*(NL+2)*(NL+2)];
    __local float3 Yl[(NL+2)*(NL+2)*(NL+2)];
    //__local float error[(NL+2)*(NL+2)*(NL+2)];
    
    float3 ki[8];
    
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int idx = iz+NZ*iy+NY*NZ*ix;
 
    unsigned int lx = get_local_id(0)+1;
    unsigned int ly = get_local_id(1)+1;
    unsigned int lz = get_local_id(2)+1;
    
    unsigned int nl = NL+2;
    unsigned int ldx = lz+nl*ly+nl*nl*lx;
    
    unsigned int gx = get_group_id(0);
    unsigned int gy = get_group_id(1);
    unsigned int gz = get_group_id(2);
    
    unsigned int gdx = gz+MZ*gy+MY*MZ*gx;
    
    bool not_border = ((ix != 0) && (ix != (NX-1)) && (iy != 0) && (iy != (NY-1)) && (iz != 0) && (iz != (NZ-1)));
    
    Yl[ldx] = X[idx];
    if ((lx == 1) && (ix != 0)) Yl[ldx-nl*nl] = X[idx-NY*NZ];
    if ((lx == nl-2) && (ix != NX-1)) Yl[ldx+nl*nl] = X[idx+NY*NZ];
    if ((ly == 1) && (iy != 0)) Yl[ldx-nl] = X[idx-NZ];
    if ((ly == nl-2) && (iy != NY-1)) Yl[ldx+nl] = X[idx+NZ];
    if ((lz == 1) && (iz != 0)) Yl[ldx-1] = X[idx-1];
    if ((lz == nl-2) && (iz != NZ-1)) Yl[ldx+1] = X[idx+1];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    ki[6] = Yl[ldx];
    
    uchar i,j;
    
    for (i = 0; i < 6; i++) {
        Xl[ldx] = X[idx];
        if ((lx == 1) && (ix != 0)) Xl[ldx-nl*nl] = X[idx-NY*NZ];
        if ((lx == nl-2) && (ix != NX-1)) Xl[ldx+nl*nl] = X[idx+NY*NZ];
        if ((ly == 1) && (iy != 0)) Xl[ldx-nl] = X[idx-NZ];
        if ((ly == nl-2) && (iy != NY-1)) Xl[ldx+nl] = X[idx+NZ];
        if ((lz == 1) && (iz != 0)) Xl[ldx-1] = X[idx-1];
        if ((lz == nl-2) && (iz != NZ-1)) Xl[ldx+1] = X[idx+1];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        ki[i] = force(Xl, DX, B, DB)*step;
        if (not_border) ki[6] += ki[i]*bi[i];
        
        ki[7] = Yl[ldx];
        for (j = 0; j <= i; j++) 
            if (not_border) ki[7] += ai[mm(i,j)]*ki[j];
        X[idx] = ki[7];
        barrier(CLK_GLOBAL_MEM_FENCE);
    } 

    Error[idx] = (pow(ki[6].x-ki[7].x,2.f)+pow(ki[6].y-ki[7].y,2.f)+pow(ki[6].z-ki[7].z,2.f))/(SCALE*SCALE);
    barrier(CLK_GLOBAL_MEM_FENCE);
}



__kernel
void Colorize(__global float3* X, __global float4* C)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);

    unsigned int idx = iz+NZ*iy+NY*NZ*ix;
    
    float absX = sqrt(X[idx].x*X[idx].x + X[idx].y*X[idx].y + X[idx].z*X[idx].z);
    
    C[idx].x = absX*10.;
    C[idx].y = 1.-absX*10.;
    C[idx].z = 0.;
    C[idx].w = 1.;
    
}
