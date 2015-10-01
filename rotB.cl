
#define NL %(block_size)d
#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
    
float3 Cross(float3 U, float3 V)
{
    float3 Y;
    Y.x = U.y*V.z - U.z*V.y;
    Y.y = U.z*V.x - U.x*V.z;
    Y.z = U.x*V.y - U.y*V.x;
    
    return Y;
}



float3 Deriv(__local float3* Xl, uchar d, uchar m)
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
    if (d == 0) {ng = NX; ii = ix; ll = lx; bl = nl*nl;}
    if (d == 1) {ng = NY; ii = iy; ll = ly; bl = nl   ;}
    if (d == 2) {ng = NZ; ii = iz; ll = lz; bl = 1    ;}
      
    if ((ii != 0) && (ii != (ng-1))) { 
        if (m == 1) out = Xl[ldx+bl]*(float3)(0.5) - Xl[ldx-bl]*(float3)(0.5);
        if (m == 2) out = Xl[ldx+bl] + Xl[ldx-bl] - (float3)(2)*Xl[ldx];              
    } else if (ii == 0) {
        if (m == 1) out = -(float3)(1.5)*Xl[ldx] + (float3)(2)*Xl[ldx+bl] - (float3)(0.5)*Xl[ldx+2*bl]; 
        if (m == 2) out = (float3)(2)*Xl[ldx] - (float3)(5)*Xl[ldx+bl] + (float3)(4)*Xl[ldx+2*bl] - Xl[ldx+3*bl];
    } else if (ii == (ng-1)) {
        if (m == 1) out = (float3)(1.5)*Xl[ldx] - (float3)(2)*Xl[ldx-bl] + (float3)(0.5)*Xl[ldx-2*bl];
        if (m == 2) out = (float3)(2)*Xl[ldx] - (float3)(5)*Xl[ldx-bl] + (float3)(4)*Xl[ldx-2*bl] - Xl[ldx-3*bl];           
    }
    return out;
}




__kernel
void Copy(__global float3* X, __global float3* Y)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    Y[idx] = X[idx];    
}


__kernel __attribute__((reqd_work_group_size(NL,NL,NL)))
void Force(__global float3* X, __global float3* J, __global float3* B, __global float3* Ix, __global float3* F)
{
    __local float3 Xl[(NL+2)*(NL+2)*(NL+2)];
    __local float3 Yl[(NL+2)*(NL+2)*(NL+2)];
    
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int idx = iz+NZ*iy+NY*NZ*ix;
 
    unsigned int lx = get_local_id(0)+1;
    unsigned int ly = get_local_id(1)+1;
    unsigned int lz = get_local_id(2)+1;
    
    unsigned int nl = NL+2;
    
    unsigned int ldx = lz+nl*ly+nl*nl*lx;
    
    float3 dX[3],dXb[3],d2X[3][3],Bx,dB[3],dBx[3];
    float Det_dX,dDet_dX[3];
    unsigned int i,j;
    
    
    //loading position vectors to local memory
    Xl[ldx] = X[idx];
    if ((lx == 1) && (ix != 0)) Xl[ldx-nl*nl] = X[idx-NY*NZ];
    if ((lx == nl-2) && (ix != NX-1)) Xl[ldx+nl*nl] = X[idx+NY*NZ];

    if ((ly == 1) && (iy != 0)) Xl[ldx-nl] = X[idx-NZ];
    if ((ly == nl-2) && (iy != NY-1)) Xl[ldx+nl] = X[idx+NZ];

    if ((lz == 1) && (iz != 0)) Xl[ldx-1] = X[idx-1];
    if ((lz == nl-2) && (iz != NZ-1)) Xl[ldx+1] = X[idx+1];
    barrier(CLK_LOCAL_MEM_FENCE); 
    
    
    //computing first and second derivatives of X
    //for (i = 0; i < 3; i++) {
    //    dX[i] = Deriv(Xl, i, 1);
    //    d2X[i][i] = Deriv(Xl, i, 2);
    //    J[idx + i*NX*NY*NZ] = dX[i];
    //}
    //barrier(CLK_LOCAL_MEM_FENCE);
    //barrier(CLK_GLOBAL_MEM_FENCE);

    for (i = 0; i < 3; i++) {
        dX[i] = Deriv(Xl, i, 1);
        d2X[i][i] = Deriv(Xl, i, 2);
        J[idx + i*NX*NY*NZ] = dX[i];
        
        //loading Jacobian to local memory
        Yl[ldx] = J[idx + i*NX*NY*NZ];
        if ((lx == 1) && (ix != 0)) Yl[ldx-nl*nl] = J[idx-NY*NZ + i*NX*NY*NZ];
        if ((lx == nl-2) && (ix != NX-1)) Yl[ldx+nl*nl] = J[idx+NY*NZ + i*NX*NY*NZ];

        if ((ly == 1) && (iy != 0)) Yl[ldx-nl] = J[idx-NZ + i*NX*NY*NZ];
        if ((ly == nl-2) && (iy != NY-1)) Yl[ldx+nl] = J[idx+NZ + i*NX*NY*NZ];

        if ((lz == 1) && (iz != 0)) Yl[ldx-1] = J[idx-1 + i*NX*NY*NZ];
        if ((lz == nl-2) && (iz != NZ-1)) Yl[ldx+1] = J[idx+1 + i*NX*NY*NZ];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        //computing cross secong derivatives of X
        for (j = 0; j < 3; j++) if (i != j) d2X[i][j] = Deriv(Yl, j, 1);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    //loading initial field components to local memory (this should be removed, since B is constant)
    Xl[ldx] = B[idx];
    if ((lx == 1) && (ix != 0)) Xl[ldx-nl*nl] = B[idx-NY*NZ];
    if ((lx == nl-2) && (ix != NX-1)) Xl[ldx+nl*nl] = B[idx+NY*NZ];

    if ((ly == 1) && (iy != 0)) Xl[ldx-nl] = B[idx-NZ];
    if ((ly == nl-2) && (iy != NY-1)) Xl[ldx+nl] = B[idx+NZ];

    if ((lz == 1) && (iz != 0)) Xl[ldx-1] = B[idx-1];
    if ((lz == nl-2) && (iz != NZ-1)) Xl[ldx+1] = B[idx+1];
    barrier(CLK_LOCAL_MEM_FENCE);
    //computing first derivatives of field
    for (i = 0; i < 3; i++) dB[i] = Deriv(Xl, i, 1);

    //the rest of code uses only private memory
    
    Det_dX = dX[0].x*dX[1].y*dX[2].z + dX[0].y*dX[1].z*dX[2].x + dX[0].z*dX[1].x*dX[2].y - 
           dX[0].x*dX[1].z*dX[2].y - dX[0].y*dX[1].x*dX[2].z - dX[0].z*dX[1].y*dX[2].x;
    
    dXb[0] = Cross(dX[1], dX[2])/(float3)(Det_dX);
    dXb[1] = Cross(dX[2], dX[0])/(float3)(Det_dX);
    dXb[2] = Cross(dX[0], dX[1])/(float3)(Det_dX);
    
    Bx.x = dX[0].x*B[idx].x + dX[1].x*B[idx].y + dX[2].x*B[idx].z;
    Bx.y = dX[0].y*B[idx].x + dX[1].y*B[idx].y + dX[2].y*B[idx].z;
    Bx.z = dX[0].z*B[idx].x + dX[1].z*B[idx].y + dX[2].z*B[idx].z;
    
    Bx /= (float3)Det_dX;
    
    for (i = 0; i < 3; i++) {
        dDet_dX[i] = 0.;   
        for (j = 0; j < 3; j++) dDet_dX[i] += d2X[i][j].x*dXb[j].x + d2X[i][j].y*dXb[j].y + d2X[i][j].z*dXb[j].z;
        dDet_dX[i] *= Det_dX;
    }
       
    for (i = 0; i < 3; i++) {
        dBx[i] = -(float3)(dDet_dX[i])*Bx;

        dBx[i].x += dX[0].x*dB[i].x + dX[1].x*dB[i].y + dX[2].x*dB[i].z;
        dBx[i].y += dX[0].y*dB[i].x + dX[1].y*dB[i].y + dX[2].y*dB[i].z;
        dBx[i].z += dX[0].z*dB[i].x + dX[1].z*dB[i].y + dX[2].z*dB[i].z;
        
        dBx[i].x += d2X[0][i].x*B[idx].x + d2X[1][i].x*B[idx].y + d2X[2][i].x*B[idx].z;
        dBx[i].y += d2X[0][i].y*B[idx].x + d2X[1][i].y*B[idx].y + d2X[2][i].y*B[idx].z;
        dBx[i].z += d2X[0][i].z*B[idx].x + d2X[1][i].z*B[idx].y + d2X[2][i].z*B[idx].z;

        dBx[i] /= (float3)(Det_dX);
    }
    
    Ix[idx] = (float3)0;
    for (i = 0; i < 3; i++) {
        Ix[idx] += Cross(dXb[i], dBx[i]);
    }
    
    F[idx] = Cross(Ix[idx], Bx);
}



__kernel
void Step(__global float3* X, __global float3* F, __global float* params, float coeff)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
//    if ((iz != 0) && (iz != (nz-1))) {    
    if ((ix != 0) && (ix != (nx-1)) && (iy != 0) && (iy != (ny-1)) && (iz != 0) && (iz != (nz-1))) {
        X[idx] += F[idx]*(float3)(params[0]*coeff);
    }
}



__kernel
void Update(__global float3* X, __global float3* Y, __global float* params)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    if (params[1] < 1.f) {
        X[idx] = Y[idx];   
    }
    
}


__kernel
void Error(__global float3* X, __global float3* Y, __global float* params)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    float err;
    
    err = ((X[idx].x-Y[idx].x)*(X[idx].x-Y[idx].x) + 
           (X[idx].y-Y[idx].y)*(X[idx].y-Y[idx].y) + 
           (X[idx].z-Y[idx].z)*(X[idx].z-Y[idx].z))/(params[2]*params[2]);
    
    if (err > params[1]) { 
        params[1] = err;
    }  
}


__kernel
void Colorize(__global float3* X, __global float4* C)
{
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
    unsigned int iz = get_global_id(2);
    
    unsigned int nx = get_global_size(0);
    unsigned int ny = get_global_size(1);
    unsigned int nz = get_global_size(2);

    unsigned int idx = iz+nz*iy+ny*nz*ix;
    
    float absX = sqrt(X[idx].x*X[idx].x + X[idx].y*X[idx].y + X[idx].z*X[idx].z);
    
    C[idx].x = absX*10.;
    C[idx].y = 1.-absX*10.;
    C[idx].z = 0.;
    C[idx].w = 1.;
    
}
